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

For the current implementation, this project supports two models. One is for debugging and simply adds a suffix "-ly" to
the end. The second model is a seq2seq encoder-decoder model. It is based off of the translation tutorial from PyTorch
docs. Instead of decoding to words, I configured this model to generate words, character by character. I believe that is
essential for creative names.

## Data
- Look on Kaggle for datasets of text that might contain business names + descriptions
- Crawl the web to get business names and descriptions for websites
- Find a place with business registrations and get business names and descriptions

#### [Update 1]
- Downloaded data from [Kaggle](https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset) to get business names
- Wrote script to query Wikipedia for business descriptions

To download the dataset from Kaggle:
- First, download your kaggle token from your account settings
- Second, run this
```bash
make download-raw-dataset
```

Once that data is downloaded, you'll need to query wikipedia for descriptions. This script takes days to run. Future work can use other ways
to query wikipedia

```bash
make query-descriptions
```

Instead of waiting for the downloader to complete, you can build an interm dataset from the existing descriptions and train with them

```bash
make build-dataset
```

At this stage, we have a dataset of 7 million business names with their descriptions. All that is left to do is to train
a model to generate a name from the description

## Exploration
Explore the distribution of words. Check if business names are usually included in the descriptions. Check length of
words and length of descriptions to make decisions on network.

After some exploration, there is high variability in the names of businesses. The higher variability is seen in the
descriptions that are pulled from wikipedia. It would have been ideal to get a one sentence elevator pitch of each
business. Such big descriptions can be a burden on one GPU and take a long time without adding much value.

## Modeling
This project seems perfect for an encoder-decoder architecture. I want to implement the easiest solution for modeling
possible to go ahead and start building the product. As a next step I could replace some parts such as encoding with
the newly trained neo-GPT.

Generating names from a "thought vector" is no easy task since the model doesn't know why it is getting punished.
The model designed here is merely a baseline. It takes in the description of a business (first 20 words of it), runs
it through a recurrent encoder block, to generate a thought vector. The thought vector is passed on to an 
attention-powered decoder that generates characters to form the business name. After several iterations, it can be seen
that the model generates the word "university" when it gets a description of one. The more iterations pass through,
the closer the generated words are to reality.

Example input-target-output
```
> oman listen oh mahn arabic uman ma n officially the sultanate of oman arabic saltanat u uman is a country
= omint
< city of riiniro<EOS>
```

Another one of universities
```
> international technological university itu is a private university in san jose california . it was founded in by 
professor shu
= international technological university
< university of sint<EOS>
```

As you can see, reading the text describing a country, or a place, generated text that resembles one. This example
also shows the noise in the data. As querying wikipedia with a business name doesn't always return that business
description.

I have had many issues with this model consuming, and not letting go of my GPU memory even after termination. I believe
it has to do with loading tensors to the GPU sentence by sentence. If this gets fixed, bigger networks could be trained.

Another phenomena that can be noticed by generating names after every 1000 iterations is that all samples follow one
path of change. Which indicates that the model is struggling with understanding the difference between the errors it
is making.

The training code also supports versioning through datetime tagging. Every model is exported as a package of language,
encoder, and decoder files. This allows the front end dashboard to access different experiments and allow for future
model improvements to be shared tot he users.

To run an experiment that trains a new model and saves it, run

```bash
make train-seq2seq
```

## Stack
### Vision
A web app that lets a user enter business descriptions, and output the generated business name

### Frontend
A streamlit dashboard that allows the user to enter text to describe their business. The output is a generated business
name.
The dashboard also allow the user to
1. Select the model type: either a heuristic simple model, or the seq-to-seq model
2. Select the model version: Using datetime tags
3. Provide feedback on the generated names

### Backend
Training and experimentation is done locally. The code for model and data handlers is shared between training and
front-end components to make code edits across the codebase easier.

Another feature that this architecture supports is modularization of the steps. I.e. for the data downloading and
preparation steps, we have independent components that can be run separately. Since they are long-running processes,
this can help us distribute the runs on different threads or even machines (or lambda functions) to build up the
dataset. Each of them reads from a folder, and writes to another folder. This pattern can also be seen with tools
such as Azure Data Factory. It allowed me to build up different components and assuming others are working well then tie
them together.

# Learnings and further work

- Poor model: The task at hand is not simple. Trying to come up with business names from humans is not an easy task
  as there are many variables that contribute to making "a good name". Figuring out a cost function is essential to
  make this model work. Another problem is the limitation of local compute. Even with RTX 3080, this model takes a long
  time to run fully. This is caused by the fact that we are using a sequential model, a GPU can only parallelize so
  much. Potential solution could be to use a transformer model to overcome that.
- Variability in dataset. Our dataset was hand constructed which meant that quality was going to be an issue. Some
  cleaning could make a big difference. However, the names and variability are still an issue. For example, what would
  a model learn from the name "ibm"?. Starting an embedding from scratch adds to this struggle. Using a pretrained SOTA
  model as the encoder could make a big difference and jump start our model.
- Using simple loaders doesn't work when we have a dataset of this size, or bigger. It quickly puts a strain on the
  memory and we miss out on the nice optimizations that are provided by packages suck as pytorch-lightning. It was a
  good learning experience to approach this problem of breaking a notebook into packages, from scratch. However, it
  put a limit on what kind of model we could develop and halted iterations severely.
- Taking the package to the cloud. It has been nice developing locally on my own GPU. However, it did postpone worrying
  about cloud challenges. Deploying a simple "hello world" app to the cloud then improving upon that as we go is a
  better approach and makes sure we always have a deployed model.
- Fault tolerance. The code that queries the dataset, one row at a time, from wikipedia is super slow. It takes days,
  to weeks, to run. One error that resulted in a NaN stopped it around 100K samples. With such long running queues,
  it is better to think about the possible errors that could come out, and implement retrying and fixing so we don't
  get halted midway.
