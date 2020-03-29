# HeavyWater Machine Learning Problem

* [Purpose](#Purpose)
* [Problem Statement](#Problem-Statement)
* [Mission](#Mission)
* [Measurement Criteria](#Measurement-Criteria)
* [A few more details](#A-few-more-details)
* [Project Setup](#Project-Setup)
* [RESTful URLs](#RESTful-URLs)

### Purpose

The purpose of this problem is to evaluate your abilities in several dimensions at once.

  1. Do you understand the principles of ML/AI/data science/<insert fancy other term here>
  1. Can you build something that works
  1. Do you have a grasp of the tool chain from code on your local to code in production
  1. Can you explain your design and thinking process
  1. Are you excited by learning and challenges


### Problem Statement

We process documents related to mortgages, aka everything that happens to originate a mortgage that you don't see as a borrower. Often times the only access to a document we have is a scan of a fax of a print out of the document. Our system is able to read and comprehend that document, turning a PDF into structured business content that our customers can act on.

This dataset represents the output of the OCR stage of our data pipeline. Since these documents are sensitive financial documents we have not provided you with the raw text that was extracted. Instead we have had to obscure the data. Each word in the source is mapped to one unique value in the output. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from our OCR layer, so it should be _roughly_ in order.

Here is a sample line:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

The first field is the document label. Everything after the comma is a space delimited set of word values.

The dataset is included as part of this repo.

### Mission

Train a document classification model. Deploy your model to a public cloud platform (AWS/Google/Azure/Heroku) as a webservice, send us an email with the URL to you github repo, the URL of your publicly deployed service so we can submit test cases and a recorded screen cast demo of your solution's UI, its code and deployment steps. Also, we use AWS so we are partial to you using that ... just saying.


### Measurement Criteria

We will measure your solution on the following criteria:

  1. Does your webservice work?
  1. Is your hosted model as accurate as ours? Better? (think confusion matrix)
  1. Your code, is it understandable, readable and/or deployable?
  1. Do you use industry best practices in training/testing/deploying?
  1. Do you use modern packages/tools in your code and deployment pipeline like [this](https://stelligent.com/2016/02/08/aws-lambda-functions-aws-codepipeline-cloudformation/)?
  1. The effectiveness of your demo, did you frame the problem and your approach to a solution, did you explain your thinking and any remaining gaps, etc?
  1. Are we able to run your testcases against your webservice? Can we run them against our webservice?


### A few more details

Webservice spec:

- RESTful API
- Respect content-type header (application/json and text/html minimum other bonus)
- Discoverable from root path
- URL encoded GET parameter "words" returns predicted document type (confidence is a bonus) in field "prediction" and "confidence"
- HTML pages should be readable by a human and allow for action, aka input field and submit buttons etc.
- Even a broken clock is right twice a day. A working webservice is a good first goal. It could return the highest likelihood doc class.



## Solution
Technologies Used
 - Python Programming Language
 - Numpy, Pandas, scikit-learn libraries for data processing and creating LSA model
 - json for saving LSA model
 - Flask for creating RESTful APIs

Note: Final Folder in Solution Holds the finished and tested files

### Project Setup

1.	To download Python: [click here](https://www.python.org/downloads/)  
	Install the software as mentioned in it, and add its path to the system environment variable

2. 	To download Pip: [click here](https://bootstrap.pypa.io/get-pip.py)  
	Save the file `ctrl+s` (file should save in .py format)  
	Open command prompt in the download location: `python get-pip.py`

3.	To install Libraries that used in the project  
	Type in the command prompt: `pip install -r requirements.txt` in root dir of the project

4.	To check if everything installed properly  
	In command prompt: `python`  
	In Python console: `import flask, numpy, pandas, json, sklearn`  
	**If you get no error, Project Setup is Done**

5.	To run the project:
	Extract shuffled-full-set-hashed.csv file in `Problem` folder in the same dir.
	Open Command Prompt and navigate to `Solution/Final` where app.py file is located  
	Type: `python app.py`
	In the web browser type `localhost:5000/` to start using the web interface of the project  

## RESTful URLs
 - '/' : GET
	To get started and to check if project is running or not.

 - '/DataRead' : GET | POST <br>
 	GET- Read the data from default location<br>
	POST- Read the data from the path given in post request<br>
		Format = { "path" : ("<data_path>")stringType }

- '/DataSplit' : GET | POST<br>
 	GET- Take default data splitting Value, Ratio of 7:3 (Training : Testing)<br>
	POST- assign new splitting ratio<br>
		Format = { "ratio" : (<splitting_ratio>)floatType }

- '/DataPrcessing' : GET<br>
 	Process the training data and create [mapper](#mapper) dictionary<br>

- '/TFIDF' : GET<br>
 	This will start TFIDF process on the training data<br>
	To Know more on TFIDF [click here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

- '/TopWords' : GET<br>
 	This will create [rareWords](#rareWords) dictionary from processed training data with the help of TFIDF API

- '/SaveLSA' : GET<br>
	It will save LSA model in the local repository, which can be used later

- '/LoadLSA' : GET | POST<br>
	GET- It will load LSA model from Default dir<br>
	POST- It will load LSA model from the path given in post request<br>
		Format = { "path" : ("<data_path>")stringType }<br>
	I highly recommend to save and load LSA model and then run new inputs for predictions on them, it will save data and TFIDF processing time.

- '/PredictOne' : POST<br>
	It will predict the output for one string statement and return {"prediction" : "<predictedOutput>"}<br>
	Format = { "loaded" : ("< True:False (use loaded LSA?:use new LSA?) >")boolType, "data" : "<input>"(stringType) }


- '/PredictMany' : POST
	It will predict the output for multiple string statements and return {"prediction" : ["<predictedOutputs>",...]}<br>
	Format = { "loaded" : ("< True:False (use loaded LSA?:use new LSA?) >")boolType, "data" : ["<inputs>"(stringType)] }


- '/Testing' : GET
	It will setup testing environment and take care of variables which are needed for getting further results.

- '/ConfusionMatrix' : GET
	return the dictionary type confusion matrix of the LSA model<br>
	Format = {<("categoryKey")stringType> : <ConfusionMatrixforEachKey( [ [00,01], [10,11] ] )>}

- '/Score' : POST
	return the dictionary type of LSA model Score<br>
	Format = {<("categoryKey")stringType> : <ScoreforEachKey( {"Accuracy": float, "Precision": float, "Recall": float, "F1_Score": float, "Support": int} )> }

- '/ScoreView' : POST
	return a string type of score of LSA model, which will give a better visual of score<br>

##### mapper
	it will map all the unique encoded words from string input to a unique ID
##### rareWords
	Top 100 most important words for each category of output, like BILL, RETURNED CHECK, POLICY CHANGE, etc.
