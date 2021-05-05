# Business Name Generator

For this project, I will be using the idea suggested in the guide, generating business names. I’ve been working on my own app and for the last year, one of the hardest things for was to find a name. There are so many requirements that I had to abide by, such as:
- Making it creative
- Catchy
- Make sure its meaning is related to what the app does
- Doesn’t already exist
- Isn’t cheesy
- Domain name isn’t taken

My favourite type of projects to work on is fixing a problem that I suffer from.
I think that some solutions exist for this task and some similar ones that I can use to jump start the modeling part. I chose this one because it will allow me to put way more attention on the other parts of the stack, i.e. data collection, web app building, monitoring. I also hope to be able to focus on getting users for the product to get the experience of real world feedback.
## Task
Generate the business name
### Options
- Generate the name character by character
- Generate the name by sampling words and concatenating them
## Data
- Look on Kaggle for datasets of text that might contain business names + descriptions
- Crawl the web to get business names and descriptions for websites
- Find a place with business registrations and get business names and descriptions

#### [Update 1]
- Downloaded data from [Kaggle](https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset) to get business names
- Wrote script to query Wikipedia for business descriptions

## Exploration
Explore the distribution of words. Check if business names are usually included in the descriptions. Check length of words and length of descriptions to make decisions on network.
## Modeling
This project seems perfect for an encoder-decoder architecture. I want to implement the easiest solution for modeling possible to go ahead and start building the product. As a next step I could replace some parts such as encoding with the newly trained neo-GPT.
## Stack
### Vision
A web app that lets a user enter business descriptions, and output the generated business name
### Frontend
A streamlit dashboard that allows the user to enter text to describe their business. The output could be few suggested names.
### Backend
Training and experimentation can be done locally. Once ready, the model can be wrapped with a FastAPI endpoint and containerized.
### Deployment
A simple deployment could be to an AWS EC2 machine. We could use some services such as AWS ECS to handle scale.
