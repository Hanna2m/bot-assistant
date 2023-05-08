import express from 'express';
import * as dotenv from 'dotenv'
import cors from 'cors'
import { Configuration, OpenAIApi } from 'openai';
import path from 'path'
import context from './getContext.js';

dotenv.config()

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
})

const openai = new OpenAIApi(configuration);

const app = express()
const __dirname = path.dirname("")
const buildPath = path.join(__dirname, "../client/")
app.use(express.static(buildPath))
app.use(cors())
app.use(express.json())

app.get('/', async (req, res) => {
  res.sendFile(
    path.join(__dirname, "../client/index.html"),
    function(err) {
      if(err) {
        res.status(500).send(err)
      }
    }
  )
})
app.post('/', async (req, res) => {
  try {
    const question = req.body.prompt
    const prompt = await context(question)
    // const queryContext = await context(question)
    // const prompt = `${question}\n${queryContext}`
      console.log(`Prompt: ${prompt}`)
      if (prompt) {
        const response = await openai.createCompletion({
          model: "text-davinci-003",
          prompt: `${prompt}`,
          temperature: 0,
          max_tokens: 2048,
          stop: '\n\n',
          n: 1,
      
        })
        console.log(`Response: ${response.data.choices[0].text}`)
        res.status(200).send({
          bot: response.data.choices[0].text
        })  
      }
      
  } catch (error) {
    console.log(error)
    res.status(500).send({ error })
  }
})

const PORT = process.env.PORT || 5000
app.listen(PORT, () => console.log(`Server is running on port ${PORT}`))
