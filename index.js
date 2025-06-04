import express from "express";
import multer from "multer";
import { PDFExtract } from "pdf.js-extract";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import * as dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const port = 3000;
app.use(cors());

const upload = multer({ storage: multer.memoryStorage() });

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "embedding-001",
});
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-1.5-flash",
  temperature: 0.1,
});

// Vector Database
const pinecone = new Pinecone({
  apiKey: process.env.VECTOR_DATABASE_API_KEY,
});
const pineconeIndex = pinecone.Index(process.env.VECTOR_DATABASE_INDEX_NAME);

//  pdf.js-extract
const pdfExtract = new PDFExtract();

app.get("/", async (req, res) => {
  res.status(200).json({ message: "Backend running successfully!" });
});

app.post("/upload-document", upload.single("document"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No document uploaded.");
  }

  // ... (inside app.post('/upload-document', ...) ) ...

  try {
    const options = {};
    const data = await pdfExtract.extractBuffer(req.file.buffer, options);
    let fullText = "";

    if (data.pages && data.pages.length > 0) {
      fullText = data.pages
        .map((page) => page.content.map((item) => item.str).join(" "))
        .join("\n\n");
    } else {
      throw new Error("No text found in PDF or PDF parsing failed.");
    }

    // --- NEW: Clean up the extracted text before chunking ---
    fullText = fullText
      .replace(/\s+/g, " ") // Replace multiple spaces/newlines with a single space
      .replace(/(\r\n|\n|\r)/gm, " ") // Replace all types of newlines with a space
      .trim(); // Trim leading/trailing whitespace
    // You might also want to remove page numbers/headers if they are consistent
    // For instance, if 'P a g e' is always there: fullText = fullText.replace(/P a g e/g, '');

    console.log(
      "Text extracted (cleaned):",
      fullText.substring(0, 200) + "..."
    );

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    // This will create new documents with cleaned pageContent
    const docs = await textSplitter.createDocuments([fullText]);
    console.log(`Split document into ${docs.length} chunks.`);

    // ... (rest of your /upload-document endpoint) ...
    const upsertRequests = docs.map((doc, index) => {
      return {
        id: `doc-${Date.now()}-${index}`,
        values: [],
        metadata: { pageContent: doc.pageContent },
      };
    });

    const chunkTexts = docs.map((doc) => doc.pageContent);
    const chunkEmbeddings = await embeddings.embedDocuments(chunkTexts);

    chunkEmbeddings.forEach((embedding, index) => {
      upsertRequests[index].values = embedding;
    });

    await pineconeIndex.upsert(upsertRequests);
    console.log(`Uploaded ${docs.length} embeddings to Pinecone.`);

    res
      .status(200)
      .json({ message: "Document processed and indexed successfully." });
  } catch (error) {
    console.error("Error processing document:", error);
    res
      .status(500)
      .json({ error: "Error processing document: " + error.message });
  }
});

app.post("/ask-question", express.json(), async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).send("Question is required.");
  }

  try {
    console.log(`[Ask Question] Incoming question: "${question}"`);

    const vectorStore = new PineconeStore(embeddings, {
      pineconeIndex,
      textKey: "pageContent",
    });
    const retriever = vectorStore.asRetriever(5);
    const questionEmbedding = await embeddings.embedQuery(question);
    const queryResponse = await pineconeIndex.query({
      vector: questionEmbedding,
      topK: 5,
      includeMetadata: true,
    });
    console.log(
      "[Ask Question] Raw Pinecone Query Response:",
      JSON.stringify(queryResponse, null, 2)
    );

    const relevantDocs = queryResponse.matches.map(
      (match) =>
        new Document({
          pageContent: match.metadata.pageContent,
          metadata: { score: match.score },
        })
    );

    console.log(
      "[Ask Question] Relevant Documents (LangChain format):",
      relevantDocs
    );
    relevantDocs.forEach((doc, i) => {
      console.log(`  Doc ${i + 1} (Score: ${doc.metadata.score}):`);
      console.log(
        `    Content (raw): "${doc.pageContent.substring(0, 300)}..."`
      );
      console.log(
        `    Content (JSON stringified): ${JSON.stringify(
          doc.pageContent.substring(0, 300) + "..."
        )}`
      );
    });

    if (relevantDocs.length === 0) {
      console.warn(
        "[Ask Question] No relevant documents found by Pinecone for this question."
      );
      return res.status(200).json({
        answer:
          "I couldn't find relevant information in the document to answer your question. Please try rephrasing or provide more context.",
      });
    }

    const contextualizedPrompt = PromptTemplate.fromTemplate(
      `You are an intelligent assistant designed to provide answers based on the *provided context only*.
    
        Carefully read the following context:
        {context}
    
        Now, please answer the user's question. Answer the question as per the context, try to be intelligent and understand the question, if the information isn't available in the provided context, politely state that you don't have enough information from the document to answer.
        Question: {question}
    
        Answer:`
    );
    const ragChain = RunnableSequence.from([
      {
        context: retriever.pipe((docsFromRetriever) => {
          console.log(
            "[Ask Question] Inside retriever.pipe - Docs received:",
            docsFromRetriever
          );

          const mappedContents = docsFromRetriever.map((doc, index) => {
            console.log(
              `[Ask Question] Inside retriever.pipe - Doc ${index} pageContent length: ${doc.pageContent.length}`
            );
            console.log(
              `[Ask Question] Inside retriever.pipe - Doc ${index} pageContent (JSON stringified): ${JSON.stringify(
                doc.pageContent.substring(0, 300) + "..."
              )}`
            );
            return doc.pageContent;
          });

          const combinedContext = mappedContents.join("\n\n");
          console.log(
            `[Ask Question] Combined Context Length: ${combinedContext.length}`
          );
          console.log(
            "[Ask Question] Combined Context fed to LLM (first 1000 chars):",
            combinedContext.substring(0, 1000) + "..."
          );
          console.log(
            `[Ask Question] Combined Context fed to LLM (JSON stringified, first 1000 chars): ${JSON.stringify(
              combinedContext.substring(0, 1000) + "..."
            )}`
          );

          return combinedContext;
        }),
        question: new RunnablePassthrough(),
      },
      contextualizedPrompt,
      model,
      new StringOutputParser(),
    ]);

    const answer = await ragChain.invoke(question);
    console.log("[Ask Question] Final Answer from LLM:", answer);

    res.status(200).json({ answer });
  } catch (error) {
    console.error("Error answering question:", error);
    res
      .status(500)
      .json({ error: "Error answering question: " + error.message });
  }
});
app.listen(port, () => {
  console.log(`Backend server listening at http://localhost:${port}`);
});
