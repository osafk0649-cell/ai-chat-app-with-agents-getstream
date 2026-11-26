import OpenAI from "openai";
import type { Channel, DefaultGenerics, Event, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";
import { OpenAIResponseHandler } from "./OpenAIResponseHandler";

export class OpenAIAgent implements AIAgent {
  private openai?: OpenAI;
  private history: OpenAI.Chat.ChatCompletionMessageParam[] = [];
  private lastInteractionTs = Date.now();

  private handlers: OpenAIResponseHandler[] = [];

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();

    this.handlers.forEach((handler) => handler.dispose());
    this.handlers = [];
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.OPENAI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("OpenAI API key is required");
    }

    this.openai = new OpenAI({
      apiKey,
      baseURL: process.env.OPENAI_BASE_URL || undefined,
    });

    this.chatClient.on("message.new", this.handleMessage);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    return `You are an expert AI Writing Assistant. Your primary purpose is to be a collaborative writing partner.

**Your Core Capabilities:**
- Content Creation, Improvement, Style Adaptation, Brainstorming, and Writing Coaching.
- **Web Search**: You have the ability to search the web for up-to-date information using the 'web_search' tool.
- **Current Date**: Today's date is ${currentDate}. Please use this for any time-sensitive queries.

**Crucial Instructions:**
1.  **ALWAYS use the 'web_search' tool when the user asks for current information, news, or facts.** Your internal knowledge is outdated.
2.  When you use the 'web_search' tool, you will receive a JSON object with search results. **You MUST base your response on the information provided in that search result.** Do not rely on your pre-existing knowledge for topics that require current information.
3.  Synthesize the information from the web search to provide a comprehensive and accurate answer. Cite sources if the results include URLs.

**Response Format:**
- Be direct and production-ready.
- Use clear formatting.
- Never begin responses with phrases like "Here's the edit:", "Here are the changes:", or similar introductory statements.
- Provide responses directly and professionally without unnecessary preambles.

**Writing Context**: ${context || "General writing assistance."}

Your goal is to provide accurate, current, and helpful written content. Failure to use web search for recent topics will result in an incorrect answer.`;
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.openai) {
      console.log("OpenAI not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    const writingTask = (e.message.custom as { writingTask?: string })
      ?.writingTask;
    const context = writingTask ? `Writing Task: ${writingTask}` : undefined;
    const instructions = this.getWritingAssistantPrompt(context);

    this.history.push({ role: "user", content: message });
    // keep memory small: last 30 messages
    if (this.history.length > 30) {
      this.history = this.history.slice(-30);
    }

    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    });

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    const handler = new OpenAIResponseHandler(
      this.openai,
      this.history,
      instructions,
      this.chatClient,
      this.channel,
      channelMessage,
      () => this.removeHandler(handler)
    );
    this.handlers.push(handler);
    void handler.run();
  };

  private removeHandler = (handlerToRemove: OpenAIResponseHandler) => {
    this.handlers = this.handlers.filter(
      (handler) => handler !== handlerToRemove
    );
  };
}