import OpenAI from "openai";
import { _iterSSEMessages } from "./utils/streaming";

// import Writer from 'writer-sdk';

// const client = new Writer({
//   baseURL: 'http://app.qordobadev.com/api/',
// });

// const stream = await client.completions.create({
//   model: 'palmyra-x-003-instruct',
//   prompt: 'Hi, my name is',
//   stream: true,
// });
// for await (const streamingData of stream) {
//   console.log(streamingData.value);
// }

/**
 * Async iterator that reads from a ReadableStream of Uint8Array,
 * decodes the stream, splits it into lines, and parses JSON messages.
 */
async function* streamDecodedIterator(
  response: Response
): AsyncIterableIterator<any> {
  // Create an AbortController to pass along.
  const controller = new AbortController();

  try {
    // Iterate over the SSE messages from the stream.
    for await (const sse of _iterSSEMessages(response, controller)) {
      // Check for termination.
      if (sse.data.startsWith("[DONE]")) {
        continue; // or return?
      }

      // If there is no event, yield the parsed JSON directly.
      if (sse.event === null) {
        let data;
        // Attempt to parse the data field.
        try {
          data = JSON.parse(sse.data);
        } catch (e) {
          console.error(`Could not parse message into JSON:`, sse.data);
          console.error(`From chunk:`, sse.raw);
          throw e;
        }

        if (data && data.error) {
          throw new Error(`SSE data error ${data.error}`);
        }
        yield data;
      } else {
        let data;
        try {
          data = JSON.parse(sse.data);
        } catch (e) {
          console.error(`Could not parse message into JSON:`, sse.data);
          console.error(`From chunk:`, sse.raw);
          throw e;
        }
        // TODO: Is this where the error should be thrown?
        if (sse.event == "error") {
          throw new Error(`SSE error ${data.message} ERROR: ${data.error}`);
        }
        // Otherwise, yield an object with the event and the parsed data.
        yield { event: sse.event, data };
      }
    }
  } catch (err) {
    console.error("Error in iterator", err);
  }
}

/**
 * Create a stream that reads from the chat completions endpoint.
 *
 * // TODO: Rewrite the function to wrap the call function similar to the following:
 *  https://github.com/langchain-ai/langchainjs/blob/main/libs/langchain-openai/src/chat_models.ts#L1798
 *
 * @param requestData - The request data to send to the endpoint.
 * @returns An async iterator that reads from the stream.
 */
export async function createChatCompletionStream(
  requestData: any
): Promise<AsyncIterable<any> | null> {
  const endpoint = "https://app.qordobadev.com/api/private/chat/completions";
  const apiKey = "t"; // dev

  // console.log(
  //   'calling createChatCompletionStream',
  //   JSON.stringify({ ...requestData, stream: true }, null, 2)
  // );

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ ...requestData, stream: true }),
    });

    if (!response.body) {
      throw new Error("No body in response");
    }

    // Wrap the ReadableStream in an async iterator.
    return streamDecodedIterator(response);
  } catch (error) {
    console.error("Error calling chat completions endpoint:", error);
    throw error;
  }
}

/**
 * Create a chat completion using the given request data.
 * @param requestData - The request data to send to the endpoint.
 * @returns The chat completion response.
 */
export async function createChatCompletion(
  requestData: any
): Promise<OpenAI.Chat.Completions.ChatCompletion> {
  // const endpoint = 'http://app.qordobadev.com/api/private/chat/completions';
  // const endpoint = this.apiBase + '/chat/completions';
  // const apiKey = 't'; // dev

  // console.log(
  //   'calling createChatCompletion',
  //   JSON.stringify(requestData, null, 2)
  // );

  try {
    const response = await fetch(
      "https://app.qordobadev.com/api/private/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Authorization: `Bearer tttt`,
          // Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({ ...requestData, stream: false }),
      }
    );

    if (!response.ok || !response.body) {
      throw new Error(
        `Failed to call Writer endpoint: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling chat completions endpoint:", error);
    throw error;
  }
}
