import type {
  ResponseFormatText,
  ResponseFormatJSONObject,
  ResponseFormatJSONSchema,
  // FunctionDefinition,
} from "openai/resources/shared";
// import { zodToJsonSchema } from "zod-to-json-schema";
import {
  AIMessage,
  AIMessageChunk,
  BaseMessageFields,
  ChatMessage,
  ChatMessageChunk,
  FunctionMessageChunk,
  HumanMessageChunk,
  isAIMessage,
  OpenAIToolCall,
  SystemMessageChunk,
  ToolMessage,
  ToolMessageChunk,
  UsageMetadata,
  type BaseMessage,
} from "@langchain/core/messages";
import {
  BaseLanguageModelInput,
  StructuredOutputMethodOptions,
  // StructuredOutputMethodParams,
  TokenUsage,
  type BaseLanguageModelCallOptions,
} from "@langchain/core/language_models/base";

import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  // BaseChatModelCallOptions,
  type BaseChatModelParams,
  BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import {
  ChatGeneration,
  ChatGenerationChunk,
  ChatResult,
} from "@langchain/core/outputs";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { ToolCallChunk } from "@langchain/core/messages/tool";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import {
  Runnable,
  // RunnablePassthrough,
  // RunnableSequence,
} from "@langchain/core/runnables";
import { formatToOpenAIToolChoice, OpenAIToolChoice } from "./utils";
import OpenAI from "openai";
import { NewTokenIndices } from "@langchain/core/callbacks/base";
import {
  convertLangChainToolCallToOpenAI,
  // JsonOutputKeyToolsParser,
  makeInvalidToolCall,
  parseToolCall,
} from "@langchain/core/output_parsers/openai_tools";
// import {
//   JsonOutputParser,
//   StructuredOutputParser,
//   type BaseLLMOutputParser,
// } from "@langchain/core/output_parsers";
import { createChatCompletion, createChatCompletionStream } from "./client";
import { z } from "zod";
import { WriterCallOptions, WriterChatInput } from "./types";
import { ChatCompletionChunk, Usage } from "./writer-sdk-types";

type ChatOpenAIResponseFormatJSONSchema = Omit<
  ResponseFormatJSONSchema,
  "json_schema"
> & {
  json_schema: Omit<ResponseFormatJSONSchema["json_schema"], "schema"> & {
    /**
     * The schema for the response format, described as a JSON Schema object
     * or a Zod object.
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    schema: Record<string, any> | z.ZodObject<any, any, any, any>;
  };
};
export type ChatOpenAIResponseFormat =
  | ResponseFormatText
  | ResponseFormatJSONObject
  | ChatOpenAIResponseFormatJSONSchema;

type OpenAIRoleEnum =
  | "system"
  | "developer"
  | "assistant"
  | "user"
  | "function"
  | "tool";

type OpenAICompletionParam = OpenAI.Chat.Completions.ChatCompletionMessageParam;

function extractGenericMessageCustomRole(message: ChatMessage) {
  if (
    message.role !== "system" &&
    message.role !== "developer" &&
    message.role !== "assistant" &&
    message.role !== "user" &&
    message.role !== "function" &&
    message.role !== "tool"
  ) {
    console.warn(`Unknown message role: ${message.role}`);
  }

  return message.role as OpenAIRoleEnum;
}
export function messageToOpenAIRole(message: BaseMessage): OpenAIRoleEnum {
  const type = message._getType();
  switch (type) {
    case "system":
      return "system";
    case "ai":
      return "assistant";
    case "human":
      return "user";
    case "function":
      return "function";
    case "tool":
      return "tool";
    case "generic": {
      if (!ChatMessage.isInstance(message))
        throw new Error("Invalid generic chat message");
      return extractGenericMessageCustomRole(message);
    }
    default:
      throw new Error(`Unknown message type: ${type}`);
  }
}

/**
 * Convert a tool choice to the format expected by OpenAI.
 * TODO: If the model doesn't support tool choices, this should throw an error.
 * @param toolChoice The tool choice to convert.
 * @returns The converted tool choice.
 **/
function _convertToOpenAITool(tool: BindToolsInput) {
  return convertToOpenAITool(tool);
}

function isReasoningModel(model?: string) {
  return model?.startsWith("reasoning-model");
}

// TODO: Use the base structured output options param in next breaking release.
export interface ChatOpenAIStructuredOutputMethodOptions<
  IncludeRaw extends boolean,
> extends StructuredOutputMethodOptions<IncludeRaw> {
  /**
   * strict: If `true` and `method` = "function_calling", model output is
   * guaranteed to exactly match the schema. If `true`, the input schema
   * will also be validated according to
   * https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
   * If `false`, input schema will not be validated and model output will not
   * be validated.
   * If `undefined`, `strict` argument will not be passed to the model.
   *
   * @version 0.2.6
   * @note Planned breaking change in version `0.3.0`:
   * `strict` will default to `true` when `method` is
   * "function_calling" as of version `0.3.0`.
   */
  strict?: boolean;
}

// Used in LangSmith, export is important here
export function _convertMessagesToOpenAIParams(
  messages: BaseMessage[],
  model?: string
): OpenAICompletionParam[] {
  // TODO: Function messages do not support array content, fix cast
  return messages.flatMap((message) => {
    let role = messageToOpenAIRole(message);
    if (role === "system" && isReasoningModel(model)) {
      role = "developer";
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const completionParam: Record<string, any> = {
      role,
      content: message.content,
    };
    if (message.name != null) {
      completionParam.name = message.name;
    }
    if (message.additional_kwargs.function_call != null) {
      completionParam.function_call = message.additional_kwargs.function_call;
      completionParam.content = null;
    }
    if (isAIMessage(message) && !!message.tool_calls?.length) {
      completionParam.tool_calls = message.tool_calls.map(
        convertLangChainToolCallToOpenAI
      );
      completionParam.content = null;
    } else {
      if (message.additional_kwargs.tool_calls != null) {
        completionParam.tool_calls = message.additional_kwargs.tool_calls;
      }
      if ((message as ToolMessage).tool_call_id != null) {
        completionParam.tool_call_id = (message as ToolMessage).tool_call_id;
      }
    }

    if (
      message.additional_kwargs.audio &&
      typeof message.additional_kwargs.audio === "object" &&
      "id" in message.additional_kwargs.audio
    ) {
      const audioMessage = {
        role: "assistant",
        audio: {
          id: message.additional_kwargs.audio.id,
        },
      };
      return [completionParam, audioMessage] as OpenAICompletionParam[];
    }

    return completionParam as OpenAICompletionParam;
  });
}

export interface ChatWriterCallOptions
  extends WriterCallOptions,
    BaseLanguageModelCallOptions {
  tools?: BindToolsInput[];
  tool_choice?: OpenAIToolChoice;
  /**
   * Additional options to pass to streamed completions.
   * If provided takes precedence over "streamUsage" set at initialization time.
   */
  stream_options?: {
    /**
     * Whether or not to include token usage in the stream.
     * If set to `true`, this will include an additional
     * chunk at the end of the stream with the token usage.
     */
    include_usage: boolean;
  };
  // Not supported yet
  // response_format?: ChatOpenAIResponseFormat;
}

type ClientOptions = { apiKey?: string; organization?: string };

/**
 * Input interface for ChatWriter
 */
export interface ChatWriterFields
  extends Partial<WriterChatInput>,
    BaseChatModelParams {
  configuration?: ClientOptions; // TODO: Actually type this see OpenAI/Writer
}

/**
 * Integration with a chat model.
 */
export class ChatWriter
  // export class ChatWriter<
  //     CallOptions extends ChatWriterCallOptions = ChatWriterCallOptions,
  //   >
  // Extend BaseLanguageModelCallOptions and pass it as the generic here
  // to support typing for additional runtime parameters for your integration
  extends BaseChatModel<ChatWriterCallOptions, AIMessageChunk>
  implements Partial<WriterChatInput>
{
  // Used for tracing, replace with the same name as your class
  static lc_name() {
    return "ChatWriter";
  }

  get callKeys() {
    return [...super.callKeys, "options", "tools", "tool_choice"];
  }

  lc_serializable = true;

  /**
   * Replace with any secrets this class passes to `super`.
   * See {@link ../../langchain-cohere/src/chat_model.ts} for
   * an example.
   */
  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: "WRITER_API_KEY",
      organization: "WRITER_ORGANIZATION",
    };
  }

  get lc_aliases(): { [key: string]: string } | undefined {
    return {
      modelName: "model",
      apiKey: "writer_api_key",
    };
  }

  apiBase = "http://app.qordobadev.com/api/private";
  apiKey?: string;
  topP?: number;
  n = 1;
  model = "palmyra-x-004";
  temperature = 0;
  version = "4.3"; // passed in meta
  organization?: string; // passed in meta
  streaming = false;
  streamUsage = false;
  maxTokens?: number;

  logprobs?: boolean;
  response_format?: ChatOpenAIResponseFormat;
  __includeRawResponse?: boolean;

  // TODO: Integrate with Writer SDK
  // protected client: WriterClient;
  protected clientConfig: ClientOptions;

  constructor(fields?: ChatWriterFields) {
    super(fields ?? {});

    this.apiKey =
      fields?.apiKey ??
      fields?.configuration?.apiKey ??
      getEnvironmentVariable("WRITER_API_KEY");

    this.organization =
      fields?.configuration?.organization ??
      getEnvironmentVariable("WRITER_ORGANIZATION") ??
      "1";

    this.model = fields?.model ?? this.model;

    this.temperature = fields?.temperature ?? this.temperature;
    this.topP = fields?.topP ?? this.topP;
    this.logprobs = fields?.logprobs;
    this.n = fields?.n ?? this.n;
    this.maxTokens = fields?.maxTokens;
    this.logprobs = fields?.logprobs;
    // this.response_format = fields?.response_format;
    this.__includeRawResponse = fields?.__includeRawResponse;

    this.streaming = fields?.streaming ?? false;
    this.streamUsage = fields?.streamUsage ?? this.streamUsage;

    this.clientConfig = {
      apiKey: this.apiKey,
      organization: this.organization,
      ...fields?.configuration,
    };
  }

  _llmType() {
    return "chat_palmyra";
  }

  /**
   * Implement to support tool calling.
   * You must also pass the bound tools into your actual chat completion call.
   * See {@link ../../langchain-cerberas/src/chat_model.ts} for
   * an example.
   */
  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<this["ParsedCallOptions"]>
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    BaseLanguageModelCallOptions
  > {
    return this.bind({
      tools: tools.map((tool) => _convertToOpenAITool(tool)),
      ...kwargs,
    });
  }

  /**
   * Get the parameters used to invoke the model
   */
  invocationParams(
    options: this["ParsedCallOptions"],
    extra?: {
      streaming?: boolean;
    }
  ) {
    let streamOptionsConfig = {};
    if (options?.stream_options !== undefined) {
      streamOptionsConfig = { stream_options: options.stream_options };
    } else if (this.streamUsage && (this.streaming || extra?.streaming)) {
      streamOptionsConfig = { stream_options: { include_usage: true } };
    }

    // TODO: Type this like... Omit<
    // OpenAIClient.Chat.ChatCompletionCreateParams,
    // "messages"> https://github.com/langchain-ai/langchainjs/blob/main/libs/langchain-openai/src/chat_models.ts#L1149C19-L1151C17
    const params = {
      model: this.model,
      temperature: this.temperature,
      top_p: this.topP,
      max_tokens: this.maxTokens,
      logprobs: this.logprobs,
      n: this.n,
      // if include_usage is set or streamUsage then stream must be set to true.
      stream: this.streaming,
      tools: options?.tools?.length
        ? options.tools.map((tool) => _convertToOpenAITool(tool))
        : undefined,
      tool_choice: formatToOpenAIToolChoice(options?.tool_choice),
      meta: {
        version: this.version,
        organizationId: this.organization,
      },
      ...streamOptionsConfig,
      // __includeRawResponse: this.__includeRawResponse,
    };

    return params;
  }

  protected _convertOpenAIChatCompletionMessageToBaseMessage(
    message: OpenAI.Chat.Completions.ChatCompletionMessage,
    rawResponse: OpenAI.Chat.Completions.ChatCompletion
  ): BaseMessage {
    const rawToolCalls: OpenAIToolCall[] | undefined = message.tool_calls as
      | OpenAIToolCall[]
      | undefined;
    switch (message.role) {
      case "assistant": {
        const toolCalls = [];
        const invalidToolCalls = [];
        for (const rawToolCall of rawToolCalls ?? []) {
          try {
            toolCalls.push(parseToolCall(rawToolCall, { returnId: true }));
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } catch (e: any) {
            invalidToolCalls.push(makeInvalidToolCall(rawToolCall, e.message));
          }
        }
        const additional_kwargs: Record<string, unknown> = {
          function_call: message.function_call,
          tool_calls: rawToolCalls,
        };
        if (this.__includeRawResponse !== undefined) {
          additional_kwargs.__raw_response = rawResponse;
        }
        const response_metadata: Record<string, unknown> | undefined = {
          model_name: rawResponse.model,
          ...(rawResponse.system_fingerprint
            ? {
                usage: { ...rawResponse.usage },
                system_fingerprint: rawResponse.system_fingerprint,
              }
            : {}),
        };

        if (message.audio) {
          additional_kwargs.audio = message.audio;
        }

        return new AIMessage({
          content: message.content || "",
          tool_calls: toolCalls,
          invalid_tool_calls: invalidToolCalls,
          additional_kwargs,
          response_metadata,
          id: rawResponse.id,
        });
      }
      default:
        return new ChatMessage(
          message.content || "",
          message.role ?? "unknown"
        );
    }
  }

  protected _convertOpenAIDeltaToBaseMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    rawResponse: ChatCompletionChunk, // TODO: Type from Writer SDK WriterClient.Chat.Completions.ChatCompletionChunk
    defaultRole?: OpenAIRoleEnum
  ) {
    const role = delta.role ?? defaultRole;
    const content = delta.content ?? "";
    let additional_kwargs: Record<string, unknown>;
    if (delta.function_call) {
      additional_kwargs = {
        function_call: delta.function_call,
      };
    } else if (delta.tool_calls) {
      additional_kwargs = {
        tool_calls: delta.tool_calls,
      };
    } else {
      additional_kwargs = {};
    }
    if (this.__includeRawResponse) {
      additional_kwargs.__raw_response = rawResponse;
    }

    if (delta.audio) {
      additional_kwargs.audio = {
        ...delta.audio,
        index: rawResponse.choices[0].index,
      };
    }

    const response_metadata = { usage: { ...rawResponse.usage } };
    if (role === "user") {
      return new HumanMessageChunk({ content, response_metadata });
    } else if (role === "assistant") {
      const toolCallChunks: ToolCallChunk[] = [];
      if (Array.isArray(delta.tool_calls)) {
        for (const rawToolCall of delta.tool_calls) {
          toolCallChunks.push({
            name: rawToolCall.function?.name,
            args: rawToolCall.function?.arguments,
            id: rawToolCall.id,
            index: rawToolCall.index,
            type: "tool_call_chunk",
          });
        }
      }
      return new AIMessageChunk({
        content,
        tool_call_chunks: toolCallChunks,
        additional_kwargs,
        id: rawResponse.id,
        response_metadata,
      });
    } else if (role === "system") {
      return new SystemMessageChunk({ content, response_metadata });
    } else if (role === "developer") {
      return new SystemMessageChunk({
        content,
        response_metadata,
        additional_kwargs: {
          __openai_role__: "developer",
        },
      });
    } else if (role === "function") {
      return new FunctionMessageChunk({
        content,
        additional_kwargs,
        name: delta.name,
        response_metadata,
      });
    } else if (role === "tool") {
      return new ToolMessageChunk({
        content,
        additional_kwargs,
        tool_call_id: delta.tool_call_id,
        response_metadata,
      });
    } else {
      return new ChatMessageChunk({ content, role, response_metadata });
    }
  }

  /** @ignore */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const usageMetadata = {} as UsageMetadata;
    const tokenUsage: TokenUsage = {};
    const request = this._getChatRequest(messages, options);
    // const messagesMapped: OpenAICompletionParam[] =
    //   _convertMessagesToOpenAIParams(messages, this.model);

    // Handle streaming
    if (request.stream) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      const finalChunks: Record<number, ChatGenerationChunk> = {};
      for await (const chunk of stream) {
        chunk.message.response_metadata = {
          ...chunk.generationInfo,
          ...chunk.message.response_metadata,
        };
        const index =
          (chunk.generationInfo as NewTokenIndices)?.completion ?? 0;
        if (finalChunks[index] === undefined) {
          finalChunks[index] = chunk;
        } else {
          finalChunks[index] = finalChunks[index].concat(chunk);
        }
      }
      const generations = Object.entries(finalChunks)
        .sort(([aKey], [bKey]) => parseInt(aKey, 10) - parseInt(bKey, 10))
        .map(([_, value]) => value);

      return { generations, llmOutput: { estimatedTokenUsage: tokenUsage } };
    }

    // Handle non-streaming
    const data = await this.caller.call(async () => {
      try {
        const response = await createChatCompletion(request);
        return response;
      } catch (e: any) {
        e.status = e.status ?? e.statusCode;
        throw e;
      }
    });

    const {
      completion_tokens: completionTokens,
      prompt_tokens: promptTokens,
      total_tokens: totalTokens,
      prompt_tokens_details: promptTokensDetails,
      completion_tokens_details: completionTokensDetails,
    } = data?.usage ?? {};

    if (completionTokens) {
      usageMetadata.output_tokens =
        (usageMetadata.output_tokens ?? 0) + completionTokens;
    }

    if (promptTokens) {
      usageMetadata.input_tokens =
        (usageMetadata.input_tokens ?? 0) + promptTokens;
    }

    if (totalTokens) {
      usageMetadata.total_tokens =
        (usageMetadata.total_tokens ?? 0) + totalTokens;
    }

    if (promptTokensDetails?.cached_tokens !== null) {
      usageMetadata.input_token_details = {
        ...(promptTokensDetails?.cached_tokens !== null && {
          cache_read: promptTokensDetails?.cached_tokens,
        }),
      };
    }

    if (completionTokensDetails?.reasoning_tokens !== null) {
      usageMetadata.output_token_details = {
        ...(completionTokensDetails?.reasoning_tokens !== null && {
          reasoning: completionTokensDetails?.reasoning_tokens,
        }),
      };
    }
    console.log("NON-STREAMING PATH");

    const generations: ChatGeneration[] = [];
    for (const part of data?.choices ?? []) {
      const text = part.message?.content ?? "";
      const generation: ChatGeneration = {
        text,
        message: this._convertOpenAIChatCompletionMessageToBaseMessage(
          part.message ?? { role: "assistant" },
          data
        ),
      };
      generation.generationInfo = {
        ...(part.finish_reason ? { finish_reason: part.finish_reason } : {}),
        ...(part.logprobs ? { logprobs: part.logprobs } : {}),
      };
      if (isAIMessage(generation.message)) {
        generation.message.usage_metadata = usageMetadata;
      }
      // Fields are not serialized unless passed to the constructor
      // Doing this ensures all fields on the message are serialized
      generation.message = new AIMessage(
        Object.fromEntries(
          Object.entries(generation.message).filter(
            ([key]) => !key.startsWith("lc_")
          )
        ) as BaseMessageFields
      );
      generations.push(generation);
    }

    return {
      generations,
      llmOutput: {
        tokenUsage: {
          promptTokens: usageMetadata.input_tokens,
          completionTokens: usageMetadata.output_tokens,
          totalTokens: usageMetadata.total_tokens,
        },
      },
    };
  }

  /**
   * Implement to support streaming.
   * Should yield chunks iteratively.
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model);
    const request = {
      ...this._getChatRequest(messages, options),
      messages: messagesMapped, // clean up messages in above request builder
      stream: true,
    };
    let defaultRole: OpenAIRoleEnum | undefined;

    // All models have a built in `this.caller` property for retries
    const stream = await this.caller.call(async () => {
      let stream;
      try {
        stream = await createChatCompletionStream(request);
      } catch (e: any) {
        e.status = e.status ?? e.statusCode;
        throw e;
      }
      return stream;
    });

    if (!stream) {
      throw new Error("No stream data in response");
    }

    let usage: Usage | undefined;
    for await (const data of stream) {
      const choice = data?.choices?.[0];

      if (!choice) {
        continue;
      }

      const { delta } = choice;
      if (!delta) {
        continue;
      }

      const chunk = this._convertOpenAIDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: 0, // TODO: Not yet supported
        // prompt: options.promptIndex ?? 0,
        completion: choice.index ?? 0,
      };
      if (typeof chunk.content !== "string") {
        console.log(
          "[WARNING]: Received non-string content from Writer. This is currently not supported."
        );
        continue;
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generationInfo: Record<string, any> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        // Only include system fingerprint in the last chunk for now
        // to avoid concatenation issues
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
      }
      if (this.logprobs) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text ?? "",
        newTokenIndices,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (usage) {
      const inputTokenDetails = {
        ...(usage.prompt_token_details?.cached_tokens !== null && {
          cache_read: usage.prompt_token_details?.cached_tokens,
        }),
      };
      const outputTokenDetails = {
        ...(usage.completion_tokens_details?.reasoning_tokens !== null && {
          reasoning: usage.completion_tokens_details?.reasoning_tokens,
        }),
      };
      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: "",
          response_metadata: {
            usage: { ...usage },
          },
          usage_metadata: {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            ...(Object.keys(inputTokenDetails).length > 0 && {
              input_token_details: inputTokenDetails,
            }),
            ...(Object.keys(outputTokenDetails).length > 0 && {
              output_token_details: outputTokenDetails,
            }),
          },
        }),
        text: "",
      });
      yield generationChunk;
    }
    if (options.signal?.aborted) {
      throw new Error("AbortError");
    }
  }

  _getChatRequest(messages: BaseMessage[], options: this["ParsedCallOptions"]) {
    const params = this.invocationParams(options);
    return {
      ...params,
      messages: messages.map((message) => ({
        role: messageToOpenAIRole(message),
        content: message.content,
      })),
    };
  }

  // protected _convertOpenAIDeltaToBaseMessageChunk(
  //   // The delta from the stream
  //   delta: Record<string, any>,
  //   // A generic raw response type (you can further type this if desired)
  //   rawResponse: { id: string; choices: Array<{ index: number }>; usage?: any },
  //   defaultRole?: string
  // ) {
  //   // Use the role from the delta or fall back to the provided default.
  //   const role = delta.role ?? defaultRole;
  //   // Ensure we always have a string.
  //   const content = delta.content ?? '';

  //   // Build any additional kwargs (for tool calls, function calls, etc.)
  //   let additional_kwargs: Record<string, unknown> = {};
  //   if (delta.function_call) {
  //     additional_kwargs = { function_call: delta.function_call };
  //   } else if (delta.tool_calls) {
  //     additional_kwargs = { tool_calls: delta.tool_calls };
  //   }
  //   // Optionally include the entire raw response.
  //   if (this.__includeRawResponse) {
  //     additional_kwargs.__raw_response = rawResponse;
  //   }
  //   // If there is audio data, include it along with the index from the first choice.
  //   if (delta.audio) {
  //     additional_kwargs.audio = {
  //       ...delta.audio,
  //       index: rawResponse.choices[0]?.index,
  //     };
  //   }

  //   // Build some response metadata from the raw response usage.
  //   const response_metadata = { usage: { ...rawResponse.usage } };

  //   // Now, based on the role, return the proper message chunk.
  //   if (role === 'user') {
  //     return new HumanMessageChunk({ content, response_metadata });
  //   } else if (role === 'assistant') {
  //     const toolCallChunks: ToolCallChunk[] = [];
  //     if (Array.isArray(delta.tool_calls)) {
  //       for (const rawToolCall of delta.tool_calls) {
  //         toolCallChunks.push({
  //           name: rawToolCall.function?.name,
  //           args: rawToolCall.function?.arguments,
  //           id: rawToolCall.id,
  //           index: rawToolCall.index,
  //           type: 'tool_call_chunk',
  //         });
  //       }
  //     }
  //     return new AIMessageChunk({
  //       content,
  //       tool_call_chunks: toolCallChunks,
  //       additional_kwargs,
  //       id: rawResponse.id,
  //       response_metadata,
  //     });
  //   } else if (role === 'system') {
  //     return new SystemMessageChunk({ content, response_metadata });
  //   } else if (role === 'developer') {
  //     return new SystemMessageChunk({
  //       content,
  //       response_metadata,
  //       additional_kwargs: {
  //         __openai_role__: 'developer',
  //       },
  //     });
  //   } else if (role === 'function') {
  //     return new FunctionMessageChunk({
  //       content,
  //       additional_kwargs,
  //       name: delta.name,
  //       response_metadata,
  //     });
  //   } else if (role === 'tool') {
  //     return new ToolMessageChunk({
  //       content,
  //       additional_kwargs,
  //       tool_call_id: delta.tool_call_id,
  //       response_metadata,
  //     });
  //   } else {
  //     return new ChatMessageChunk({ content, role, response_metadata });
  //   }
  // }

  /** @ignore */
  _combineLLMOutput() {
    return [];
  }

  // withStructuredOutput<
  //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //   RunOutput extends Record<string, any> = Record<string, any>
  // >(
  //   outputSchema:
  //     | z.ZodType<RunOutput>
  //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //     | Record<string, any>,
  //   config?: ChatOpenAIStructuredOutputMethodOptions<false>
  // ): Runnable<BaseLanguageModelInput, RunOutput>;

  // withStructuredOutput<
  //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //   RunOutput extends Record<string, any> = Record<string, any>
  // >(
  //   outputSchema:
  //     | z.ZodType<RunOutput>
  //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //     | Record<string, any>,
  //   config?: ChatOpenAIStructuredOutputMethodOptions<true>
  // ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  // withStructuredOutput<
  //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //   RunOutput extends Record<string, any> = Record<string, any>
  // >(
  //   outputSchema:
  //     | z.ZodType<RunOutput>
  //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //     | Record<string, any>,
  //   config?: ChatOpenAIStructuredOutputMethodOptions<boolean>
  // ):
  //   | Runnable<BaseLanguageModelInput, RunOutput>
  //   | Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  // withStructuredOutput<
  //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //   RunOutput extends Record<string, any> = Record<string, any>
  // >(
  //   outputSchema:
  //     | z.ZodType<RunOutput>
  //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //     | Record<string, any>,
  //   config?: ChatOpenAIStructuredOutputMethodOptions<boolean>
  // ):
  //   | Runnable<BaseLanguageModelInput, RunOutput>
  //   | Runnable<
  //       BaseLanguageModelInput,
  //       { raw: BaseMessage; parsed: RunOutput }
  //     > {
  //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //   let schema: z.ZodType<RunOutput> | Record<string, any>;
  //   let name;
  //   let method;
  //   let includeRaw;
  //   if (isStructuredOutputMethodParams(outputSchema)) {
  //     schema = outputSchema.schema;
  //     name = outputSchema.name;
  //     method = outputSchema.method;
  //     includeRaw = outputSchema.includeRaw;
  //   } else {
  //     schema = outputSchema;
  //     name = config?.name;
  //     method = config?.method;
  //     includeRaw = config?.includeRaw;
  //   }
  //   let llm: Runnable<BaseLanguageModelInput>;
  //   let outputParser: BaseLLMOutputParser<RunOutput>;

  //   if (config?.strict !== undefined && method === 'jsonMode') {
  //     throw new Error(
  //       "Argument `strict` is only supported for `method` = 'function_calling'"
  //     );
  //   }

  //   if (
  //     !this.model.startsWith('gpt-3') &&
  //     !this.model.startsWith('gpt-4-') &&
  //     this.model !== 'gpt-4'
  //   ) {
  //     if (method === undefined) {
  //       method = 'jsonSchema';
  //     }
  //   } else if (method === 'jsonSchema') {
  //     console.warn(
  //       `[WARNING]: JSON Schema is not supported for model "${this.model}". Falling back to tool calling.`
  //     );
  //   }

  //   if (method === 'jsonMode') {
  //     llm = this.bind({
  //       response_format: { type: 'json_object' },
  //     } as Partial<CallOptions>);
  //     if (isZodSchema(schema)) {
  //       outputParser = StructuredOutputParser.fromZodSchema(schema);
  //     } else {
  //       outputParser = new JsonOutputParser<RunOutput>();
  //     }
  //   } else if (method === 'jsonSchema') {
  //     llm = this.bind({
  //       response_format: {
  //         type: 'json_schema',
  //         json_schema: {
  //           name: name ?? 'extract',
  //           description: schema.description,
  //           schema,
  //           strict: config?.strict,
  //         },
  //       },
  //     } as Partial<CallOptions>);
  //     if (isZodSchema(schema)) {
  //       outputParser = StructuredOutputParser.fromZodSchema(schema);
  //     } else {
  //       outputParser = new JsonOutputParser<RunOutput>();
  //     }
  //   } else {
  //     let functionName = name ?? 'extract';
  //     // Is function calling
  //     if (isZodSchema(schema)) {
  //       const asJsonSchema = zodToJsonSchema(schema);
  //       llm = this.bind({
  //         tools: [
  //           {
  //             type: 'function' as const,
  //             function: {
  //               name: functionName,
  //               description: asJsonSchema.description,
  //               parameters: asJsonSchema,
  //             },
  //           },
  //         ],
  //         tool_choice: {
  //           type: 'function' as const,
  //           function: {
  //             name: functionName,
  //           },
  //         },
  //         // Do not pass `strict` argument to OpenAI if `config.strict` is undefined
  //         ...(config?.strict !== undefined ? { strict: config.strict } : {}),
  //       } as Partial<CallOptions>);
  //       outputParser = new JsonOutputKeyToolsParser({
  //         returnSingle: true,
  //         keyName: functionName,
  //         zodSchema: schema,
  //       });
  //     } else {
  //       let openAIFunctionDefinition: FunctionDefinition;
  //       if (
  //         typeof schema.name === 'string' &&
  //         typeof schema.parameters === 'object' &&
  //         schema.parameters != null
  //       ) {
  //         openAIFunctionDefinition = schema as FunctionDefinition;
  //         functionName = schema.name;
  //       } else {
  //         functionName = schema.title ?? functionName;
  //         openAIFunctionDefinition = {
  //           name: functionName,
  //           description: schema.description ?? '',
  //           parameters: schema,
  //         };
  //       }
  //       llm = this.bind({
  //         tools: [
  //           {
  //             type: 'function' as const,
  //             function: openAIFunctionDefinition,
  //           },
  //         ],
  //         tool_choice: {
  //           type: 'function' as const,
  //           function: {
  //             name: functionName,
  //           },
  //         },
  //         // Do not pass `strict` argument to OpenAI if `config.strict` is undefined
  //         ...(config?.strict !== undefined ? { strict: config.strict } : {}),
  //       } as Partial<CallOptions>);
  //       outputParser = new JsonOutputKeyToolsParser<RunOutput>({
  //         returnSingle: true,
  //         keyName: functionName,
  //       });
  //     }
  //   }

  //   if (!includeRaw) {
  //     return llm.pipe(outputParser) as Runnable<
  //       BaseLanguageModelInput,
  //       RunOutput
  //     >;
  //   }

  //   const parserAssign = RunnablePassthrough.assign({
  //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
  //     parsed: (input: any, config) => outputParser.invoke(input.raw, config),
  //   });
  //   const parserNone = RunnablePassthrough.assign({
  //     parsed: () => null,
  //   });
  //   const parsedWithFallback = parserAssign.withFallbacks({
  //     fallbacks: [parserNone],
  //   });
  //   return RunnableSequence.from<
  //     BaseLanguageModelInput,
  //     { raw: BaseMessage; parsed: RunOutput }
  //   >([
  //     {
  //       raw: llm,
  //     },
  //     parsedWithFallback,
  //   ]);
  // }
}

// function isZodSchema<
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   RunOutput extends Record<string, any> = Record<string, any>
// >(
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   input: z.ZodType<RunOutput> | Record<string, any>
// ): input is z.ZodType<RunOutput> {
//   // Check for a characteristic method of Zod schemas
//   return typeof (input as z.ZodType<RunOutput>)?.parse === 'function';
// }

// function isStructuredOutputMethodParams(
//   x: unknown
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
// ): x is StructuredOutputMethodParams<Record<string, any>> {
//   return (
//     x !== undefined &&
//     // eslint-disable-next-line @typescript-eslint/no-explicit-any
//     typeof (x as StructuredOutputMethodParams<Record<string, any>>).schema ===
//       'object'
//   );
// }
