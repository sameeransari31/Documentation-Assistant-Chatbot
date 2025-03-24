# DocuChat - Documentation Assistant 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-^1.29-red.svg)](https://streamlit.io/)

A smart documentation assistant powered by RAG (Retrieval-Augmented Generation) that helps answer questions about popular technical documentation using Groq's LLMs.

## Features ✨

- **Multi-Document Support**: Query across 6 technical documentations simultaneously
- **Session Management**: Unique session ID for each user with separate chat histories
- **Model Selection**: Choose between Gemma2-9b-It, Llama3-8b-8192, or Mixtral-8x7b-32768
- **Customizable Parameters**: Adjust temperature and max tokens for responses
- **Persistent Chat Interface**: Maintains conversation history during session
- **Efficient Caching**: Fast document loading with Streamlit caching
- **Enhanced UI**: Clean and intuitive user interface with responsive design

## Supported Documentation 📚
- Mistral AI
- React
- Next.js
- Tailwind CSS
- LangChain
