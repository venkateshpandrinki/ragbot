import fs from "fs";
import Papa from "papaparse";
import { embedMany } from "ai";
import { google } from "@ai-sdk/google";
import { db } from "@/lib/db";
import { embeddings } from "@/lib/db/schema/embeddings";
import { nanoid } from "@/lib/utils";

const BATCH_SIZE = 50;
const model = google.textEmbeddingModel("text-embedding-004");

type GitaRow = {
  ID?: string;
  Chapter?: string;
  Verse?: string;
  Shloka?: string;
  EngMeaning?: string;
};

async function main() {
  console.log("Loading CSV...");
  const csvFile = fs.readFileSync("./data/gita.csv", "utf8");
  const parsed = Papa.parse<GitaRow>(csvFile, { header: true }).data;

  console.log(`Loaded ${parsed.length} rows from CSV.`);

  // Check already inserted content
  const existing = await db.select({ content: embeddings.content }).from(embeddings);
  const existingSet = new Set(existing.map(e => e.content));

  // Filter only new rows
  const newRows = parsed.filter((row) => {
    const text = formatVerse(row);
    return text && !existingSet.has(text);
  });

  console.log(`Processing ${newRows.length} new rows for embeddings...`);

  for (let i = 0; i < newRows.length; i += BATCH_SIZE) {
    const batch = newRows.slice(i, i + BATCH_SIZE);
    const values = batch.map(row => formatVerse(row) as string); // now always string

    console.log(
      `Embedding batch ${i / BATCH_SIZE + 1}/${Math.ceil(newRows.length / BATCH_SIZE)}...`
    );

    const { embeddings: vectors } = await embedMany({
      model,
      values,
    });

    const rowsToInsert = batch.map((row, idx) => ({
      id: nanoid(),
      content: values[idx], // string only
      embedding: vectors[idx], // number[]
    }));

    await db.insert(embeddings).values(rowsToInsert);

    console.log(` Inserted ${rowsToInsert.length} rows into DB`);
  }

  console.log(" All embeddings inserted!");
}

function formatVerse(row: GitaRow): string | null {
  const chapter = row.Chapter?.trim();
  const verse = row.Verse?.trim();
  const shloka = row.Shloka?.trim();
  const meaning = row.EngMeaning?.trim();
  if (!shloka || !meaning) return null;
  return `Chapter ${chapter}, Verse ${verse}:\n${shloka}\n${meaning}`;
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
