import bot from './assets/photo.png'
import user from './assets/user.svg'
import axios from 'axios'

const form = document.querySelector('form')
const chatContainer = document.querySelector('#chat_container')

let loadInterval

function loader(element) {
  element.textContent = '';

  loadInterval = setInterval(() => {
    element.textContent += '.'

    if (element.textContent == '....') {
      console.log(element.textContent)
      element.textContent = ''
    }
  }, 300)

}

function typeText(element, text) {
  let index = 0

  let interval = setInterval(() => {
    if (index < text.length) {
      element.innerHTML += text.charAt(index);
      index++
    } else {
      clearInterval(interval)
    }
  }, 20)
}

function generateUniqueId() {
  const timestamp = Date.now();
  const xecadecimalString = Math.random().toString(16)

  return `id-${timestamp}-${xecadecimalString}`
}

function chatStripe (isAi, value, uniqueId) {
  console.log(isAi, value, uniqueId)
  return(
    `
    <div class='wrapper ${isAi && 'ai'}'>
      <div class='chat'>
        <div class='profile'>
          <img 
            src='${isAi ? bot : user}'
            alt='${isAi ? 'bot' : 'user'}'
          />
        </div>
        <div class='message' id=${uniqueId}>${value}</div>
      </div>
    </div>
  `
  )
  
}

const handleSubmit = async(e) => {
  e.preventDefault()

  const container = document.getElementById('profile')
  container.style.display = 'none'
  // const containerClass = container.classList
  // console.log(containerClass)
  // if ( container.className === 'show') {
  //   containerClass.replace('show', 'hidden')
  //   console.log(container.className)
  // }

  const data = new FormData(form)
  //user's chatStrip
  chatContainer.innerHTML += chatStripe(false, data.get('prompt'))
  form.reset()

  //bot's chatStrip
  const uniqueId = generateUniqueId();
  console.log(chatStripe(true, '', uniqueId))
  chatContainer.innerHTML += chatStripe(true, '', uniqueId);
  chatContainer.scrollTop = chatContainer.scrollHeight

  const messageDiv = document.getElementById(uniqueId)
  console.log(messageDiv)
  loader(messageDiv)

  //fetch data from server -> bot's response
  const response = await axios.post('http://localhost:5000/', {
    prompt: data.get('prompt')
  })
  .then((res) => {
    return res
  })
  .catch((error) => {
    console.log(error)
    return error
  })

  clearInterval(loadInterval)
  messageDiv.innerHTML = ''
  console.log(response)
  if (response.statusText === 'OK') {
    const data = await response.data
    const parsedData = data.bot.trim()
    console.log(parsedData)
    typeText(messageDiv, parsedData)
  } else {
    const error = await response
    messageDiv.innerHTML = 'Something went wrong'
    alert(error)
  }
  
}

form.addEventListener('submit', handleSubmit)
form.addEventListener('keyup', (e) => {
  if(e.keyCode === 13) {
    handleSubmit(e)
  }
})
