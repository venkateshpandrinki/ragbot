import {embedMany,embed} from 'ai';
import {google} from '@ai-sdk/google'
import {db} from '../db';
import { cosineDistance,desc,gt,sql } from 'drizzle-orm';
import { embeddings } from '../db/schema/embeddings';
// const embeddingModel = google.textEmbeddingModel('gemini-embedding-exp-03-07'); 
const model = google.textEmbeddingModel('text-embedding-004')



const generateChunks = (input: string): string[] => {
    return input
      .trim()
      .split('.')
      .filter(i => i !== '');
  };


  export const generateEmbeddings = async (
    value: string,
  ): Promise<Array<{ embedding: number[]; content: string }>> => {
    const chunks = generateChunks(value);
    const { embeddings } = await embedMany({
      model,
      values: chunks,
       // @ts-ignore - Ignore TypeScript errors for the next line
    // providerOptions: {
    //   google: {
    //    outputDimensionality: 1536, // Specify desired dimension if needed/possible
    //     taskType: 'RETRIEVAL_QUERY'
    //   },
    // }
    });
    console.log(embeddings[0].length)

    return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
  };
  

  
  export const generateEmbedding = async (value:string): Promise<number[]> =>{
    const input = value.replaceAll('\\n',' ');
    const {embedding}=  await embed ({
      model,
      value:input,
      //@ts-ignore
  //        providerOptions: {
  //   google: {
  //     outputDimensionality: 1536, // Specify desired dimension if needed/possible
  //       taskType: 'RETRIEVAL_QUERY'
  //   },
  // },
    });
    console.log(embedding.length)
    return embedding;
  };

export const findRelevantContent = async (userQuery: string) => {
  const userQueryEmbedded = await generateEmbedding(userQuery);

  const similarity = sql<number>`1 - (${cosineDistance(
    embeddings.embedding,
    userQueryEmbedded
  )})`;

  const similarGuides = await db
    .select({
      content: embeddings.content,
      similarity,
    })
    .from(embeddings)
    .where(gt(similarity, 0.3))
    .orderBy((t) => desc(t.similarity))
    .limit(4);
    console.log('Query embedding length:', userQueryEmbedded.length)

  // Ensure it's safe for JSON
  const results = similarGuides.map((r) => ({
    content: r.content,
    similarity: Number(r.similarity),
  }));

  console.log(
    "DEBUG - findRelevantContent result:",
    JSON.stringify(results, null, 2)
  );

  return { results };
};
