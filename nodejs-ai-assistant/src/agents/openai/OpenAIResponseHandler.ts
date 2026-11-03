import OpenAI from "openai";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

const TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "web_search",
      description:
        "Search the web for current information, news, facts, or research on any topic",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query to find information about",
          },
        },
        required: ["query"],
      },
    },
  },
];

export class OpenAIResponseHandler {
  private message_text = "";
  private is_done = false;
  private last_update_time = 0;
  private abortController = new AbortController();

  constructor(
    private readonly openai: OpenAI,
    private readonly history: OpenAI.Chat.ChatCompletionMessageParam[],
    private readonly instructions: string,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: () => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async () => {
    const { cid, id: message_id } = this.message;

    try {
      // Loop so the model can call web_search and then answer.
      for (let round = 0; round < 5; round++) {
        const stream = await this.openai.chat.completions.create(
          {
            model: MODEL,
            temperature: 0.7,
            stream: true,
            tools: TOOLS,
            messages: [
              { role: "system", content: this.instructions },
              ...this.history,
            ],
          },
          { signal: this.abortController.signal }
        );

        let content = "";
        const toolCalls: { id: string; name: string; args: string }[] = [];
        let finishReason: string | null = null;
        let announcedGenerating = false;

        for await (const chunk of stream) {
          const choice = chunk.choices[0];
          if (!choice) continue;
          const delta = choice.delta;

          if (delta.content) {
            if (!announcedGenerating) {
              announcedGenerating = true;
              await this.channel.sendEvent({
                type: "ai_indicator.update",
                ai_state: "AI_STATE_GENERATING",
                cid,
                message_id,
              });
            }
            content += delta.content;
            this.message_text += delta.content;
            const now = Date.now();
            if (now - this.last_update_time > 1000) {
              this.chatClient.partialUpdateMessage(message_id, {
                set: { text: this.message_text },
              });
              this.last_update_time = now;
            }
          }

          for (const tc of delta.tool_calls ?? []) {
            const i = tc.index;
            toolCalls[i] ??= { id: "", name: "", args: "" };
            if (tc.id) toolCalls[i].id = tc.id;
            if (tc.function?.name) toolCalls[i].name += tc.function.name;
            if (tc.function?.arguments) toolCalls[i].args += tc.function.arguments;
          }

          if (choice.finish_reason) finishReason = choice.finish_reason;
        }

        if (finishReason === "tool_calls" && toolCalls.length > 0) {
          this.history.push({
            role: "assistant",
            content: content || null,
            tool_calls: toolCalls.map((t) => ({
              id: t.id,
              type: "function" as const,
              function: { name: t.name, arguments: t.args },
            })),
          });

          await this.channel.sendEvent({
            type: "ai_indicator.update",
            ai_state: "AI_STATE_EXTERNAL_SOURCES",
            cid,
            message_id,
          });

          for (const t of toolCalls) {
            let output: string;
            if (t.name === "web_search") {
              try {
                const args = JSON.parse(t.args);
                output = await this.performWebSearch(args.query);
              } catch (e) {
                console.error("Error performing web search", e);
                output = JSON.stringify({ error: "failed to call tool" });
              }
            } else {
              output = JSON.stringify({ error: "unknown tool" });
            }
            this.history.push({
              role: "tool",
              tool_call_id: t.id,
              content: output,
            });
          }
          continue; // let the model answer using the search results
        }

        // Final answer

        
        this.history.push({ role: "assistant", content });
        await this.chatClient.partialUpdateMessage(message_id, {
          set: { text: content },
        });
        await this.channel.sendEvent({
          type: "ai_indicator.clear",
          cid,
          message_id,
        });
        break;
      }
    } catch (error) {
      if (this.abortController.signal.aborted) return;
      console.error("An error occurred during the run:", error);
      await this.handleError(error as Error);
    } finally {
      await this.dispose();
    }
  };

  dispose = async () => {
    if (this.is_done) {
      return;
    }
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose();
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) {
      return;
    }

    console.log("Stop generating for message", this.message.id);
    this.abortController.abort();

    if (this.message_text) {
      await this.chatClient.partialUpdateMessage(this.message.id, {
        set: { text: this.message_text },
      });
    }
    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) {
      return;
    }
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? "Error generating the message",
        message: error.toString(),
      },
    });
    await this.dispose();
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };
}