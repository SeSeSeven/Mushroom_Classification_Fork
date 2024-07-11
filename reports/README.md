layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked to.

For both functions to work you mustn't rename anything. The script has two dependencies that can be installed with

```bash
pip install click markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.

### Week 1

* [×] Create a git repository
* [×] Make sure that all team members have write access to the GitHub repository
* [×] Create a dedicated environment for you project to keep track of your packages
* [×] Create the initial file structure using cookiecutter
* [×] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [×] Add a model file and a training script and get that running
* [×] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [×] Remember to comply with good coding practices (`pep8`) while doing the project
* [×] Do a bit of code typing and remember to document essential parts of your code
* [×] Setup version control for your data or part of your data
* [×] Construct one or multiple docker files for your code
* [×] Build the docker files locally and make sure they work as intended
* [×] Write one or multiple configurations files for your experiments
* [×] Used Hydra to load the configurations and manage your hyperparameters
* [×] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [×] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [×] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [×] Write unit tests related to the data part of your code
* [×] Write unit tests related to model construction and or model training
* [×] Calculate the coverage.
* [×] Get some continuous integration running on the GitHub repository
* [×] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [×] Create a trigger workflow for automatically building your docker images
* [×] Get your model training in GCP using either the Engine or Vertex AI
* [×] Create a FastAPI application that can do inference using your model
* [×] If applicable, consider deploying the model locally using torchserve
* [×] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [×] Check how robust your model is towards data drifting
* [×] Setup monitoring for the system telemetry of your deployed model
* [×] Setup monitoring for the performance of your deployed model
* [×] If applicable, play around with distributed data loading
* [×] If applicable, play around with distributed model training
* [×] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [×] Revisit your initial project description. Did the project turn out as you wanted?
* [×] Make sure all group members have a understanding about all parts of the project
* [×] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- question 1 fill here ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> 12846293, 19110558, 12803283
>
> Answer:

--- question 2 fill here ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Answer: We used the third-party framework [`timm`](https://huggingface.co/docs/timm/index) (PyTorch Image Models) framework, specifically the resnet50.a1_in1k model, for our project. timm's seamless integration with PyTorch made it easy to customize and fine-tune the resnet50.a1_in1k model according to our specific requirements. It provided us with a comprehensive library of pre-trained models and a variety of image processing tools, enabling us to complete it efficiently and achieve goood results. Additionally, we used W&B to handle hyperparameters and visualize training process.

The dataset chosen for this project is the :mushroom: [Mushroom Image dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) :mushroom: on Kaggle. It comprises 6714 images across 9 different mushroom genuses:
|  | Agaricus | Amanita | Boletus | Cortinarius | Entoloma | Hygrocybe | Lactarius | Russula | Suillus |
|-|----------|---------|---------|-------------|----------|-----------|-----------|---------|---------|
| Count | 353 | 750 | 1073 | 836 | 364 | 316 | 1563 | 1148 | 311 |

--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Answer:
We used conda, docker, and git to manage our dependencies. We created a conda environment to make sure that the dependencies of our project do not cross-contaminate with others. The packages required can be found in the requirements.txt and requirements_test.txt file

To get a complete copy of our development environment, one could:
```
git clone https://github.com/cxzhang4/Mushroom_Classification.git
cd Mushroom_Classification
conda create -n myenv python=3.12
pip install -r requirements.txt
pip install -r requirements_test.txt
dvc pull
python setup.py install
```
or simply build a docker container using our dockerfile:
```
docker build -f mushroom.dockerfile . -t mushroom:latest.
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Recommended answer length: 100-200 words
>
> Answer:
We used cookiecutter template for our project but also did some changes. 

- `mushroom_classification` is rename of src for cookiecutter template, so that we use flat-layout. This contains code for the project:
      - `./config` contains .yaml files for hydra ans sweep parameter settings.
      - `./data` containss code for processing raw data.
      - `./models` contains code for building out model, trainging and prediction.
      - `./visualization` contains code for visualizing prediction results.

- `data` contains subfolder *raw* and *processed*, which can be automatically build by *dvc pull*.

- `tests` contains test files for *pytest*.

- `models` contains example of our trained model.

- `outputs` contains example of our testing results.

- `reports` contains example of our testing visualization under *figure*, reports for *coverage*, and our final report for this project.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> 

Answer: We used:
      - MyPy: static type checks. Help catch type-related errors early in the development process.
      - ruff: PEP8 styling checks.
      - pytest: automated testing. Verify the correctness of our code.

Why matters:
      - Maintainability: Consistent code is easier to read and understand, reducing time for new team members and making it simpler to track down, fix bugs and maintain.
      - Collaboration: Adhering to a common set of style and quality rules minimizes conflicts and misunderstandings among team members.
      - Reliability: Static type checking and automated tests catch errors early in the development cycle, improving the overall reliability of the code.
      - Scalability: Maintaining high standards of code quality and formatting also helps system's growth management and stability.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> 
Answer:
We have 6 tests: 
1. Test if raw dataset exists. It ensures that the necessary data is available before any processing steps are carried out.
2. Test data splitting process. 
      - Creates dummy data directories and files.
      - Runs the data splitting function and checks if the train, validation, and test directories are created correctly and populated with the expected files.
3. Tests about MushroomClassifier model. 
      - Verifies that the model instance is correctly created.
      - Tests the forward pass of the model with dummy data to ensure it produces outputs of the correct shape.
      - Checks the training and validation steps to ensure they return tensor losses.
      - Verifies the configuration of the optimizer to ensure it is correctly instantiated.
4. Test model training process.
      - Configures data transformations and training parameters.
      - Runs the train_model function and checks if the model checkpoint is saved correctly.
5. Test prediction process.
      - Runs the predict function and checks if the predictions, true labels, and metrics are generated and saved correctly.
      - Ensures the output files are created and the predictions are consistent with the true labels.
6. Test visualization process.
      - Runs the visualize_predictions function and checks if the visualization files (predictions and metrics) are created correctly.


### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> 
Answer:
The total code coverage of code is 54%, which includes all our source code. 

Even having a code coverage of 100% does not guarantee that the code is error-free, it only indicate that each line of code has been executed, not that every possible input, edge case, or scenario has been tested. 
- It does not take into account the quality of the tests or the correctness of the code logic. Even if all lines of code are executed, there may still be bugs or errors present in the code. 
- Complex interactions between components or external systems may not be adequately simulated in tests, leading to potential errors in real-world usage that are not captured during testing. 
- Code coverage does not account for edge cases or unexpected inputs that may cause the code to fail. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> 
Answer:
Yes, our workflow included using branches and pull requests. Each team member worked on a separate branch for initial coding to ensure that individual contributions were isolated and did not interfere with others' work. During our meetings, we reviewed each other's changes and merged the results. With the first constrauction of basic but complete structures, we merged the work into the main branch. Then, after each modification, we checked the changes together, created pull requests, and conducted a thorough review before merging them into the main branch.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> 
Answer:
We did use DVC in our project, and it went through an evolutionary process. Initially, we used Google Drive as remote storage, and later migrated to a GCP bucket. Although we set up data versioning, our dataset did not actually change throughout the project.

Nevertheless, the use of DVC still brought several important benefits: First, it simplified the data synchronization process between team members, ensuring that everyone was using the same version of the dataset. Second, it prepared for future data updates, and if the dataset needed to be modified or extended, we already had the appropriate infrastructure. In addition, DVC seamlessly integrated with our code version control, making the reproducibility of experiments greatly improved. Finally, by separating large data files from the code repository, we significantly improved the performance and efficiency of the version control system.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> 
Answer:
Our continuous integration setup with GitHub Actions is structured to enforce high code quality and streamline our development processes. Key components include linting, Docker image building, and multi-environment testing.

Linting: We utilize tools like flake8 to enforce Python coding standards, ensuring consistent code style and catching potential errors early in the development cycle.

Docker Image Building: Our pipeline includes steps to build Docker images, ensuring that our application is packaged correctly and ready for deployment across different environments.

Multi-Environment Testing: We test compatibility with various Python versions to ensure consistent performance and functionality across diverse setups.

Caching: To optimize build times, we implement caching for dependencies and build artifacts. This reduces redundant installations and speeds up our CI runs, enhancing overall workflow efficiency.

In conclusion, our setup with GitHub Actions integrates linting, Docker image building, and multi-environment testing to maintain code quality, ensure deployment readiness, and optimize development efficiency through effective caching strategies.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> 
Answer:
We configured our experiments using config files, which allow us to manage and modify the parameters of our experiments easily without changing the code for training session. `hydra.yaml` and `sweep.yaml` save parameters for hydra and W&B sweeping. Additionally, we used the typer library to create a command-line interface (CLI) for running our experiments, which made it easier to handle different configurations and experiment setups.
```
# make data
python mushroom_classification/data/make_dataset.py --raw_dir 'data/raw' --processed_dir 'data/processed' --val_size 0.15 --test_size 0.15 --random_state 42
# train
# see mushroom_classification/config
# predict
python mushroom_classification/models/predict_model.py --processed_dir 'data/processed' --save_model models/resnet50.ckpt --model_name resnet50.a1_in1k --num_classes 9 --batch_size 32 --output_dir outputs/

# visualize
python mushroom_classification/visualization/visualize.py --processed_dir data/processed --prediction_path outputs/predictions.npy --report_dir reports/figures --num_images 16 --figure_arrange '(4,4)' --random_state 42 --metrics_path outputs/metrics.csv
```


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> 
Answer:
- Config Files:
      We used YAML configuration files to set and store all hyperparameters and settings for our experiments. This included parameters like batch size, learning rate etc. By doing so, we ensured that anyone could easily see and understand which parameters were used for training.

- Version Control with Git:
      We tracked changes to our configuration files and code using Git. By committing changes between experiments, we maintained a log of the parameters and settings used for each run in the Git commit history. This practice ensured that all experiments were logged and could be revisited.

- Random Seed for Reproducibility:
      We included a random seed in our code to ensure that experiments could be exactly reproduced. Setting a fixed seed meant that data splits and model initialization remained consistent across runs, leading to identical results when using the same hyperparameters.

- Docker for Environment Consistency:
      We created Docker images to encapsulate our entire development environment. This ensured that our models could be run on any computer with the same setup, eliminating discrepancies due to different system configurations.

- Experiment Tracking with W&B:
      We used W&B to log various metrics during our experiments, such as training and validation loss. This provided a visual and detailed record of how our models performed over time. W&B also stored the hyperparameters for each run, making it easy to track and compare different experiments.

- Parameter Values in Makefile:
      We set parameter values in a Makefile for functions using the typer library. This made it straightforward to run experiments with predefined settings, ensuring consistency and ease of use. By defining these parameters in the Makefile, we could execute complex commands with simple make commands.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:
As seen in the first screenshot [this figure](figures/wandb1.png), we tracked the validation loss and epochs across various hyperparameter sweeps. This allowed us to analyze how different configurations impacted model performance. By monitoring validation loss, we aimed to identify the parameter combinations that resulted in the lowest loss, indicating optimal model performance.

In the second screenshot [this figure](figures/wandb2.png), we focused on tracking parameter importance. This metric helps us understand which hyperparameters have the most significant impact on model outcomes. This analysis guides us in prioritizing hyperparameter tuning efforts effectively, ensuring resources are allocated where they can yield the most substantial improvements in model performance.

Lastly, in the third screenshot [this figure](figures/wandb3.png), we explored the effects of different combinations of learning rates, batch sizes, and epochs on validation loss. This comprehensive analysis allowed us to fine-tune these critical parameters for optimal model convergence and accuracy.

These metrics are crucial for optimizing model training and deployment strategies, ensuring that our machine learning models perform effectively in various scenarios. They provide insights into how different factors influence model outcomes, guiding iterative improvements and informed decision-making throughout the project lifecycle.

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

In our project setup, we utilized Dockerfiles to set up the environment and run necessary build processes. The Dockerfile was configured to install dependencies and execute a makefile for building the application inside the container. This allowed us to create consistent development environments locally using `docker build` commands.

For automated builds and deployment, we integrated with cloud-based CI/CD pipelines. These pipelines were triggered automatically upon pushing code changes. They initiated the build process using cloud build services, ensuring that our Docker images were updated and pushed to a container registry. This streamlined approach facilitated seamless updates and deployments across different environments.

Moreover, we leveraged Vertex AI for running our containerized applications in production. By deploying our Docker images on Vertex AI, we benefited from managed services for scalability, monitoring, and efficient resource utilization, ensuring reliable performance and operational efficiency. This setup enabled us to maintain a reliable and scalable deployment process while leveraging cloud infrastructure for enhanced capabilities.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer:
Debugging:
- Print Statements
      We frequently used print() statements to trace the execution flow and inspect variable values at different points in the code. This helped us quickly identify where the code was deviating from expected behavior.
- Loggin
      We used the logging module to create more structured and persistent logs. This was particularly useful for long-running experiments, where we needed to track progress and identify issues that occurred over time.
- GitHub Copilot
      We utilized this to assist with code suggestions and error handling.
- Online Resources
      We referred to online documentation, forums like Stack Overflow, and resources to get insights and suggestions on resolving specific errors or optimizing our code.

Profiling: ***[to be finished] ???***

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

In our project, we utilized several Google Cloud Platform (GCP) services:

Cloud Storage (Bucket): Used for scalable object storage, ideal for storing static assets and large data sets with global accessibility.

Compute Engine: Provides virtual machines (VMs) for running applications and scalable workloads, offering flexibility in machine types and configurations.

Cloud Build: A CI/CD platform that automates build, test, and deployment processes, ensuring consistent and efficient software delivery pipelines.

Vertex AI: A unified platform for machine learning (ML) and AI tasks, facilitating model training, deployment, and management at scale.

Cloud Run: A serverless platform for deploying containerized applications, automatically scaling based on traffic to manage microservices and APIs efficiently.

Artifact Registry: Used for storing and managing container images and artifacts securely. It integrates with other Google Cloud services, providing versioning and access control for artifacts.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> 
Answer:
In our project, we utilized Google Cloud Platform's Compute Engine extensively, deploying instances specifically in the Europe-West region (Belgium). This choice ensured our applications and services were hosted closer to our target audience in Europe, optimizing latency and improving overall performance.

For our Compute Engine instances, we selected VMs configured under the N1 Standard machine type with SSD storage. These VMs provided the necessary computational power and storage capacity to support various aspects of our workload, including data processing, application hosting, and backend services.

By leveraging Compute Engine in the Europe-West region, we benefited from Google Cloud's robust infrastructure and reliability, ensuring high availability and efficient resource utilization. This deployment strategy helped us meet both performance and compliance requirements while effectively managing our cloud resources for optimal results in our project.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:
You can take inspiration from [this figure](figures/Cloud_Storage.png).

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP artifact registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:
You can take inspiration from [this figure](figures/Artifact_Registry.png).

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:
You can take inspiration from [this figure](figures/Cloud_Build_HIstory.png).

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:
Unfortunately, we encountered challenges while attempting to deploy our model using Google Cloud Run and FastAPI. Despite our efforts, the deployment process did not succeed due to issues such as configuration complexities or compatibility issues with our model and the deployment environment.

We faced obstacles in containerizing our model effectively and ensuring it ran seamlessly on Cloud Run. These challenges may have stemmed from dependencies, runtime configurations, or specific requirements of our model that were not fully addressed during deployment.

However, we are actively working to resolve these issues by refining our containerization strategy, addressing dependencies, and ensuring compatibility with the Cloud Run environment. Once resolved, we plan to deploy our model to Cloud Run, leveraging FastAPI for efficient API development and deployment. This approach will enable us to invoke our deployed service via HTTP requests, providing scalable and reliable model inference capabilities in a serverless environment.

***[need fastapi] ***

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did implement monitoring for our deployed model. Using WandB, we logged various metrics and runtime resources during model inference. This allowed us to track performance metrics such as accuracy, latency, and resource utilization over time. Additionally, we utilized Google Cloud's logging capabilities to store logs in Cloud Storage buckets, providing insights into application behavior and operational status.

Monitoring plays a crucial role in ensuring the longevity of our application. It enables us to detect anomalies, identify performance bottlenecks, and optimize resource allocation. With continuous monitoring, we can make data-driven decisions to improve reliability, scalability, and user experience. This proactive approach helps maintain high application availability and responsiveness, ensuring our deployed model operates efficiently and meets performance expectations over its lifecycle.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Recommended answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

During the project, we used a total of 1.41 credits. Carson used 0.4 credits, and Yina used 0.4 credits. The service costing the most was Google Cloud Platform's Compute Engine due to extensive model training and deployment activities.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

| Step 1 | Step 2 | Step 3 | Step 4 |
|--------|--------|--------|--------|
| Local Development Environment | GitHub Repository | CI/CD Pipeline | Cloud Server |
| Data Collection & Preprocessing | Model Training | Model Evaluation | Model Deployment |
| | | | Web Application Interface |

Our system architecture begins with the local development environment, where we integrate code for data processing, model training, and evaluation. Whenever we commit code and push to GitHub, it automatically triggers our Continuous Integration and Continuous Deployment (CI/CD) pipeline. The CI/CD process includes code quality checks, automated testing, and building Docker images.

From GitHub, the code is deployed to our cloud server. In the cloud, we utilize large-scale computing resources for model training and optimization. Once trained, the model is evaluated and then deployed to the production environment. Finally, we've developed a web application interface that allows users to upload mushroom images and receive automatic classification results.

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

In our project, we encountered several substantial challenges that required dedicated effort and strategic adjustments to overcome. Initially, achieving satisfactory model accuracy was a significant hurdle. We invested considerable time in fine-tuning hyperparameters, adjusting model architectures, and optimizing training procedures to enhance performance. This iterative process involved rigorous experimentation and meticulous monitoring of results to identify the most effective configurations.

Another notable challenge involved data management with Google Cloud Storage (GCS). We faced initial difficulties with data loading and saving operations within GCS buckets. To resolve this issue, we refactored our code to streamline data handling processes, ensuring smooth integration and efficient utilization of GCS for storage and retrieval tasks.

Optimizing hyperparameter sweeps also presented challenges due to extended runtime and resource consumption. To address this, we implemented strategies such as limiting the maximum number of concurrent runs per sweep and optimizing workload distribution. These adjustments helped maintain efficiency while exploring a diverse range of parameter combinations.

Additionally, integrating and configuring tools like Weights & Biases (WandB) proved challenging initially, particularly with setting up the WandB API within Docker environments. We overcame this hurdle by adopting secure environment variables (secrets) to simplify API configuration and ensure seamless integration with our workflow for experiment tracking and analysis.

Throughout the project, effective communication and collaboration among team members were essential. Regular discussions and shared insights facilitated brainstorming of solutions and prompt implementation of necessary adjustments. By leveraging expertise and maintaining a structured approach to problem-solving, we successfully navigated these challenges, ensuring progress towards achieving our project goals effectively and efficiently.



### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer:
Ziyu contributed to model development, Dockerfile creation, writing unit tests related to data parts of the code, and calculating coverage.

Ziming participated in model development, setting up DVC (Data Version Control), creating GCS (Google Cloud Storage) buckets, logging with WandB, writing hyperparameter sweeps, implementing Hydra for configuration management, setting up trigger workflows for automated Docker image builds, and writting reports.

Yina contributed to setting up DVC for data versioning, creating GCS buckets for data storage, writing project reports, and configuring trigger workflows for automated Docker image builds.

Collectively, all team members collaborated on:
- Creating the Git repository and initializing the project structure.
- Using Cookiecutter to set up the initial file structure.
- Writing configuration files specific to the project.
- Compiling dependencies into the requirements.txt file based on project needs.

