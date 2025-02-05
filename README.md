## RAG QA system for AWS Sagemaker

In this project, I have implemented Retrieval-augmented generation (RAG) to answer questions related to AWS Sagemaker. The pretrained model is flan-t5-xl, and the code was run using Sagemaker endpoints. For our prompt contexts, a csv file with questions and answers related to sagemaker can be found here: s3://jumpstart-cache-prod-us-east-2/training-datasets/Amazon_SageMaker_FAQs/

I used Pinecone to implement RAG, and used streamlit to have an interactive front-end for my deployed RAG model. Here is a screenshot of what that looks like:

As a reference, the flan model without RAG outputs "Amazon Sagemaker Edge Manager is a free service that allows you to manage your accounts", which is very vague and more imporantly incorrect (AWS services are generally not free)

<img width="1512" alt="Screenshot 2025-02-05 at 3 34 54 AM" src="https://github.com/user-attachments/assets/51a10532-1f78-4b1c-8b4b-c1edd8ee612f" />
