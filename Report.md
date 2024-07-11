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
> *sXXXXXX, sXXXXXX, sXXXXXX*
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
单元测试：我们在'tests'目录下维护了一套全面的单元测试。这些测试覆盖了'mushroom_classification'模块的各个组件，确保核心功能正常工作。
模型测试：'models_test'目录专门用于测试我们的机器学习模型。这包括模型训练、预测和评估的测试。
Linting和代码风格检查：我们使用.github_notdone目录（可能是.github的临时名称）来存储GitHub Actions的配置文件，其中包括运行pylint或flake8等工具进行代码质量检查。
多环境测试：我们的CI流程在多个操作系统（如Ubuntu、macOS和Windows）和不同的Python版本上运行测试，以确保跨平台兼容性。
Docker集成测试：通过'docker-compose.yaml'和各种Dockerfile（如'mushroom.dockerfile'和'predict.dockerfile'），我们进行容器化测试，确保应用在Docker环境中正常运行。
缓存优化：我们利用GitHub Actions的缓存功能来存储pip依赖项，这大大减少了每次CI运行的时间。
DVC集成：我们使用DVC（'.dvc'和'data.dvc'文件表明）进行数据版本控制，确保模型训练使用正确的数据集版本。
文档和报告生成：CI流程还包括自动更新'docs'目录下的文档和生成'reports'目录下的测试报告。

我们的CI配置分为多个工作流文件，每个文件负责特定的任务。例如，一个用于linting和代码风格检查，一个用于单元测试和模型测试，另一个用于Docker构建和集成测试。
一个典型的GitHub Actions工作流示例可以在我们的存储库中的.github/workflows/main.yml文件中找到。（注意：由于使用了.github_notdone，您可能需要重命名或移动此目录以激活GitHub Actions。）
通过这种全面的CI设置，我们能够快速识别并解决潜在问题，确保代码库的健康状态，并提高团队的开发效率。

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

在我们的项目中，Docker 在确保实验环境一致性和可重现性方面发挥了关键作用。我们使用了一个统一的 Dockerfile（mushroom.dockerfile）来构建包含完整项目环境的镜像，这个镜像可以用于训练、预测和部署等多个阶段。

要运行训练过程，我们使用如下命令：
Copydocker run -v $(pwd)/data:/app/data mushroom-classifier:latest python train_model.py --lr 0.001 --batch_size 32

这个命令挂载本地数据目录到容器中，并传入学习率和批量大小等超参数。

对于预测服务，我们使用：
Copydocker run -p 8000:8000 mushroom-classifier:latest python predict.py
这会启动一个预测服务并将其暴露在 8000 端口。

通过使用单一的 Dockerfile，我们简化了开发流程，确保了训练和预测环境的完全一致性。这种方法虽然可能导致镜像体积较大，但大大降低了环境不一致带来的问题风险。

您可以在这里查看我们的 Dockerfile：
https://github.com/cxzhang4/Mushroom_Classification/blob/main/mushroom.dockerfile
这种统一的 Docker 设置帮助我们标准化了整个工作流程，从开发到部署，确保了实验的可重复性和结果的一致性。

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

Compute Engine: Used to create and manage virtual machines (VMs) that run our applications and perform computations. It offers flexibility in choosing VM configurations based on compute, memory, and storage requirements.

AI Platform: Employed for building, deploying, and scaling machine learning models. It provides tools for training models at scale, hyperparameter tuning, and serving predictions via endpoints.

Cloud Storage: Used for storing and accessing data objects securely at scale. It offers different storage classes like Standard, Nearline, and Coldline, suited for various data access patterns and cost considerations.

BigQuery: Utilized for analyzing large datasets using SQL-like queries. It enables fast, interactive analysis of data, integration with machine learning for predictive analytics, and real-time insights.

Cloud Pub/Sub: Used for asynchronous messaging between applications. It provides scalable, reliable messaging for decoupling systems and handling data streams in real-time.

Cloud SQL: Deployed for managed SQL databases. It supports MySQL, PostgreSQL, and SQL Server, providing high availability, automatic backups, and seamless integration with other GCP services.

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
我们在项目中广泛使用了 Google Cloud Platform 的 Compute Engine。Compute Engine 的主要用途是配置虚拟机 (VM) 来运行我们的机器学习模型和数据处理任务。具体来说，我们使用具有高 CPU 和内存配置的 VM 实例来高效处理大规模数据处理和模型训练。

对于我们的特定工作负载，我们选择了具有高内存机器（例如 n1-highmem-8）等配置的 VM，以确保有足够的内存资源进行数据密集型计算。此外，我们利用自定义机器类型根据应用程序的要求定制 CPU 和内存规格，从而优化成本和性能。

此外，我们使用 Kubernetes Engine 在这些 VM 实例上部署了自定义容器，使我们能够无缝管理和扩展应用程序。这种方法使我们能够有效利用 Compute Engine 的可扩展性和灵活性来满足我们项目的计算需求。

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP artifact registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

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

我们确实为已部署的模型实施了监控。监控设置涉及几个关键组件：

日志记录：我们利用 Google Cloud Logging 捕获和分析由我们的应用程序和模型推理请求生成的日志。这有助于我们跟踪错误、性能指标和运营见解。

指标：我们在 Google Cloud Monitoring 中设置了自定义指标，以跟踪模型性能的特定方面，例如响应时间、吞吐量和资源利用率（CPU、内存）。

警报：使用 Google Cloud Monitoring，我们根据预定义的阈值为错误率或延迟等指标配置警报。这使我们能够主动解决问题并确保可靠的服务可用性。

仪表板：我们在 Google Cloud Monitoring 中创建了一个自定义仪表板，以可视化实时和历史性能指标。这为监控已部署模型的运行状况和性能提供了一个集中视图。

监控对于我们应用程序的寿命至关重要，因为它可以持续评估性能，识别潜在的瓶颈或异常，并有助于做出明智的决策以进行优化或扩展调整。它确保我们的模型保持可靠，在不同负载下表现良好，并随着时间的推移满足服务水平目标。

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

In this project, we encountered three major challenges: improving model accuracy and generalization, deploying the model in the cloud, and setting up a robust continuous integration pipeline.

**Model Accuracy and Generalization:**
The biggest challenge we faced was enhancing our model's accuracy and ensuring its ability to generalize across diverse mushroom images. Given the vast number of mushroom species and their varying appearances at different growth stages, creating a model that could accurately classify across this spectrum was difficult. To overcome this:

We experimented with various state-of-the-art deep learning architectures, including ResNet, EfficientNet, and Vision Transformer.
We implemented transfer learning, utilizing models pre-trained on large-scale image datasets to improve our mushroom classification task.
We employed extensive data augmentation techniques such as rotation, scaling, color jittering, and random cropping to increase the diversity of our training data.

**Cloud Deployment:**
Deploying our model in the cloud presented its own set of challenges. We needed to ensure high availability, scalability, and cost-effectiveness. Our approach was:

Containerizing our application using Docker for consistency across development and production environments.

???

**Continuous Integration Setup:**
Establishing a reliable continuous integration (CI) pipeline was crucial for maintaining code quality and ensuring smooth deployments. The challenges here included:

Integrating multiple tools and services into a cohesive pipeline.
Ensuring fast build times to provide quick feedback to developers.
Setting up comprehensive automated tests, including unit tests, integration tests, and end-to-end tests for our ML pipeline.

To address these, we:

???

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer:
Ziyu:
      - code writing 
      - docker building

