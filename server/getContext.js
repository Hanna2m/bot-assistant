import { readFileSync, writeFileSync } from 'fs'
import { Configuration, OpenAIApi } from 'openai';
import tf from '@tensorflow/tfjs-node'
import use from '@tensorflow-models/universal-sentence-encoder'
import * as dotenv from 'dotenv'

import { encode } from 'gpt-3-encoder';

dotenv.config()

const EMBED_MODEL = "text-embedding-ada-002"
const MAX_TOKENS = 1800
const SEPARATOR = "\n* "
const separatorLenght = encode(SEPARATOR)
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
})
const openai = new OpenAIApi(configuration);
openai.api_key = process.env.OPENAI_API_KEY;

//Using use model
async function createEmbeddings(text) {
  console.log('embedding model start')
  const model = await use.load();
  const embeddings = await model.embed(text)
  return embeddings.arraySync()[0]
}

// Path to the file containing the text data
const documentPath = 'personal_content.txt';
const text = readFileSync(documentPath, 'utf-8')
const paragraphs = text.split('\n').map(paragraph => paragraph.trim()).filter(Boolean);


// const paragraphEmbeddings = await openai.createEmbedding({
//   model: EMBED_MODEL,
//   input: paragraphs
// })

// async function saveEmbeddingsToJSON(paragraphs, filename) {
//   const json = {};
//   const paragraphEmbeddings = await openai.createEmbedding({
//     model: EMBED_MODEL,
//     input: paragraphs
//   })
//   .then(res => {
//     return res.data.data[0].embedding
//   })

//   for (let i = 0; i < paragraphs.length; i++) {
//     json[paragraphs[i]] = paragraphEmbeddings[i];
//   }
//   const jsonString = JSON.stringify(json);
//   writeFileSync(filename, jsonString);
// }

const filename = './embeddings.json';
// saveEmbeddingsToJSON(paragraphs, filename)

const readContextEmbeddings = (filename) => {
  const jsonString = readFileSync(filename)
  const json = JSON.parse(jsonString)
  return json
}
const contextEmbeddings = readContextEmbeddings(filename)

const context = async (query) => {
  
  const vectorSimilarity = (x,y) => {
    return x.reduce((acc, curr, i) => acc + curr * y, 0);
  }

  const similarities = (queryEmbedding, contextEmbeddings) => {
    // Calculate similarity of query embedding with document embeddings in contexts
    let documentSimilarities = Object.entries(contextEmbeddings).map(([docIndex, docEmbedding]) => {
      let similarity = vectorSimilarity(queryEmbedding, docEmbedding);
      return [similarity, docIndex];
    });
    // Sort document similarities in descending order and return the result
    return documentSimilarities.sort(([similarity1], [similarity2]) => similarity2 - similarity1);
  }

  // const queryEmbeddings = await createEmbeddings(query)

  // Using openai api model
    const queryEmbeddings = await openai.createEmbedding({
      model: EMBED_MODEL,
      input: query
      })
    .then(res => {
      return res.data.data[0].embedding
    })
  const similarity = similarities(queryEmbeddings, contextEmbeddings).map(function(value,index) { return value[1]; }) 

  const chosenSections = []
  let chosenSectionLength = 0
  let selectedContext = ''

  for (let i = 0; i < similarity.length; i++ ) {
    chosenSectionLength += encode(similarity[i]).length
    console.log(`chosenSectionLength: ${chosenSectionLength}`)
    if (chosenSectionLength > MAX_TOKENS) {
      break
    } else {
      chosenSections.push((similarity[i]))
    }
    selectedContext = chosenSections.join(' ')
  }
  const constructPrompt = (question, context) => {
    const header = `Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know. I can only construct a response based on text contained in Sasha’s knowledge base and I can't find an answer.".`
    const promptText = header + context + "\n\n Q: " + question + "\n A:"
    const prompt = context
    return promptText
    }

  return constructPrompt(query, selectedContext)
}

export default context

