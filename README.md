# Technology forecasting using GNN 

Research for master's degree of data scienece
<br></br>

## Autonomous technology forecasting with GNN 
### About project
In this project, I used link prediction algorithm based graph neural network to predict promising technology at self-driving vehicle field. I compared two different GNN models, `graph convolutional network` and `variational graph auto-encoder`. Among them, variational graph auto-encoder performs better than GCN. So I conduct link prediction task using VGAE in this project.

I will upload my paper ASAP. 

Please check [this](https://github.com/Kiminjo/Technology-forecasting-using-GNN/files/7453594/2021._.pdf) if you want to know, how to make co-contribution network and how to extract promising network in network. 

I will upload presentation file about link prediction.
<br></br>

### Experiments details and results

The framework of this project is as follows.

![](https://user-images.githubusercontent.com/42087965/139810746-f9ee70e6-2311-472b-911f-da859d40b051.png)

A network was built based on a 'co-contribution relationship' between repositories. This is like projecting a heterogeneous network of developer-repositories to developers.

Refer to the figure below for how to build the network.

![image](https://user-images.githubusercontent.com/42087965/139811294-0e13ed86-a85f-414c-a83f-9a141f7d3c2f.png)


Community detection (Louvaion method) was used to create a community in the network, which represents an independent research area in the field of autonomous driving open source. In this study, six current major technical fields were derived.

The figure below represents six major autonomous driving open source technologies at the present time. Each node represents a repository. Through this, you can know the main technologies at the moment and the main repositories for each technology.

![major tech](https://user-images.githubusercontent.com/42087965/139811793-99babdda-173e-482f-9e78-3bd12517af3f.png)

The figure below shows the result of running community detection again after link prediction. Through this, promising technologies for autonomous driving open source in the future can be derived.

![promising tech](https://user-images.githubusercontent.com/42087965/139812075-9b00558f-6fdf-4400-ad25-6f168d79745c.png)


<br></br>
<br></br>

## Dataset

Studies on prediction of promising technologies in the past have mostly used paper data. However, the paper data has a disadvantage in that it is difficult to discover the latest research trends due to the time it takes from research to registration. So, I would like to use open source data to solve such shortcomings and make predictions about promising technologies that reflect the latest research trends.

The data used in the project are 385 repositories including keywords related to 'autonomous driving'. 

Each repository has basic information such as 'repository name', 'owner', and 'star counts' as well as data such as 'contributor list'.
<br></br>

### Statistics

- 23,017 repositories contain related keywords such as 'self-driving car' or 'autonomous drivig'
  
- 3.2% repositories are owned by 'organization' not 'user'. n this study, only repositories owned by these 'organizations' are dealt with.

- 385 repositories remained after filtering by 'contributor conts', 'stargazer couns' and 'forker counts'. They are finally used in experiments.
<br></br>

### Features 
|data        |data type|
|:---:        |:---:|
|repository name|str|
|repository ID|int|
|owner ID|int|
|owner type|str|
|repository full name | str|
|topcis|list|
|contributors|list|
|contributor counts|int|
|stargazer counts|int|
|forker counts|int|
|created date|date|
|last updated datae|date|
|readme|str|

<br></br>
<br></br>

## Software Requirements

- python >= 3.5
- pytorch >= 1.9
- pytorch geometric >= 2.02 : There are methods that are not supported in lower versions, so be sure to install them in this version or higher. Typically, the 'Train test edge split' method is not supported in previous versions. 
- scikit-learn
- numpy 
- pandas 
- scipy 
- gephi : Tools for network visualization. It is not necessary to use this, but in this project, network visualization was performed using gephi. See [here](https://gephi.org/) for more details.

<br></br>
<br></br>

## Key files 

- `link_prediction_GCN.py` : Conduct link prediction using graph convolutional network model. In this project this model was not used because it did not perform well compared to other models.

- `link_prediction_GAE.py` : The model used to predict the actual link. It gave better performance compared to GCN.

- `utils.py` : Files are included to build the network and visualize the results. If you want to check to the degree or centrality of the network, run this file.
