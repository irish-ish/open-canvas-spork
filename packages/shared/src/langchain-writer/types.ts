import { BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";

// TODO use Writer.Core.RequestOptions from the writer SDK is possible
export type WriterCoreRequestOptions<
  Req extends object = Record<string, unknown>,
> = {
  path?: string;
  query?: Req | undefined;
  body?: Req | undefined;
  headers?: Record<string, string | null | undefined> | undefined;

  maxRetries?: number;
  stream?: boolean | undefined;
  timeout?: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  httpAgent?: any;
  signal?: AbortSignal | undefined | null;
  idempotencyKey?: string;
};

export interface WriterCallOptions extends BaseLanguageModelCallOptions {
  /**
   * Additional options to pass to the underlying axios request.
   * Can this still work with fetch??
   */
  options?: WriterCoreRequestOptions;
}

export declare interface WriterBaseInput {
  /**
   * Sampling temperature.
   * Defaults to 0 if not specified.
   */
  temperature?: number;

  /**
   * Maximum number of tokens to generate in the completion. -1 returns as many
   * tokens as possible given the prompt and the model's maximum context size.
   */
  maxTokens?: number;

  /**
   * Maximum number of tokens to generate in the completion. -1 returns as many
   * tokens as possible given the prompt and the model's maximum context size.
   * Alias for `maxTokens` for reasoning models.
   */
  // maxCompletionTokens?: number;

  /** Total probability mass of tokens to consider at each step */
  topP: number;

  /** Number of completions to generate for each prompt */
  n: number;

  /** Unique string identifier representing your end-user, which can help OpenAI to monitor and detect abuse. */
  user?: string;

  /** Whether to stream the results or not. Enabling disables tokenUsage reporting */
  streaming: boolean;

  /**
   * Whether or not to include token usage data in streamed chunks.
   * @default true
   */
  streamUsage?: boolean;

  /**
   * Writer model name, e.g. "palmyra-x-004".
   * Defaults to "palmyra-x-004" if not specified.
   */
  model?: string;

  /**
   * Timeout to use when making requests to Writer
   */
  timeout?: number;

  /**
   * API key to use when making requests to Writer. Defaults to the value of
   * `WRITER_API_KEY` environment variable.
   */
  apiKey?: string;
}

export interface WriterChatInput extends WriterBaseInput {
  /**
   * Whether to return log probabilities of the output tokens or not.
   * If true, returns the log probabilities of each output token returned in the content of message.
   */
  logprobs?: boolean;

  /**
   * An integer between 0 and 5 specifying the number of most likely tokens to return at each token position,
   * each with an associated log probability. logprobs must be set to true if this parameter is used.
   */
  // topLogprobs?: number;

  /**
   * Whether to include the raw OpenAI response in the output message's "additional_kwargs" field.
   * Currently in experimental beta.
   */
  __includeRawResponse?: boolean;

  /**
   * Whether the model supports the `strict` argument when passing in tools.
   * If `undefined` the `strict` argument will not be passed to OpenAI.
   */
  supportsStrictToolCalling?: boolean;
}
