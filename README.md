# Global Wheat Head🌾 Detection🕵️‍♂️ 
![wheat](https://user-images.githubusercontent.com/64481847/92102906-db006880-edfc-11ea-805b-e14c8e669b48.png)

### **Model OUTPUT** glimpse ♾️:

![wheat](https://user-images.githubusercontent.com/64481847/92202310-52d1a000-ee9c-11ea-8a8e-77b02e845561.gif)

**Reach out <a href="https://github.com/Adk2001tech/Global-Wheat-Head-Detection/tree/master/wheat_head_video/Video_sample_Predicted">here</a> for more videos!**




# Abstract : 
Wheat is a staple across the globe, which makes this project more accountable.

**Detection of wheat heads** is an important task allowing to estimate pertinent traits including head population density and head characteristics such as sanitary state, size, maturity stage and the presence of awns. Project aims to analyze and detect wheat heads from outdoor images of wheat plants around the globe🌐 by leveraging the power of **Computer Vision** on pretty large **image(with its annotation)** dataset. 



This project is a part of <a href="https://www.kaggle.com/c/global-wheat-detection">**Global Wheat Detection**</a> challenge hosted on Kaggle.




# Dataset 🗃️:

The Global Wheat Head Dataset is led by nine research institutes from seven countries: *University of Tokyo*, *Institut national de recherche pour l’agriculture*, *l’alimentation et l’environnement*, *Arvalis*, *ETHZ*, *University of Saskatchewan*, *University of Queensland*, *Nanjing Agricultural University*, and *Rothamsted Research*. These institutions are joined by many in their pursuit of accurate wheat head detection, including the *Global Institute for Food Security*, *DigitAg*, *Kubota*, and *Hiphen*.

<img src="https://user-images.githubusercontent.com/64481847/92110293-19e7eb80-ee08-11ea-88e8-a4f533444cb9.png" height="200px" weidth="550">


Using worldwide data, the DL model will focus on a generalized solution of detecting the wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the *training dataset covers multiple regions*. 3,000 images from Europe (France, UK, Switzerland) and North America (Canada). The test data includes about 1,000 images from Australia, Japan, and China. Data can be accessible from below links:

* <a href="https://arxiv.org/abs/2005.02162">**Global Wheat Head Detection (GWHD) dataset**</a>
* <a href="https://www.kaggle.com/c/global-wheat-detection/data">**Global Wheat Detection Data**</a>

<img src="https://user-images.githubusercontent.com/64481847/92114952-4f440780-ee0f-11ea-9278-b59059a0408b.png" height="500px" weidth="1550px">


**Pre-Processed Images with Bounding Boxes**

<img src="https://user-images.githubusercontent.com/64481847/92112876-148ca000-ee0c-11ea-9ce6-8c30c7d63ad5.jpg" height="450px" weidth="1700px">

### It encapsulate:
* `Train.csv` of shape `(147793, 5)` with following columns:

      1. image_id - the unique image ID(3,373)
      2. width, height - the width and height of the images
      3. bbox - a bounding box, formatted as a Python-style list of [xmin, ymin, width, height]
      4. source
      
* 3,422 *training images* with given 10 *test images*.
* Rest *test images* are hidden.


# Exploratory Data Analysis 📊:
**Notebook link:**
* <a href="https://www.kaggle.com/akhileshdkapse/global-wheat-detection-comprehensive-eda">**Kaggle Notebook**</a>
* <a href="https://github.com/Adk2001tech/Global-Wheat-Head-Detection/blob/master/Wheat-Head-detection-comprehensive-eda.ipynb">**GitHub Repository Notebook**</a>

In order to avoid unnecessary computation during the Exploratory Data Analysis(EDA) that follows, here we will prepare custom dataframe encapsulating necessary image oriented features such as :
* `image_id` - The unique image ID(3,373).
* `width, height` - The width and height of the images.
* `source` - Image belonging source tags.
* `boxes` - All bounding box per unique image, formatted as a Python-style list of [xmin, ymin, width, height].
* `box_count` - Totall Bounding box count per unique image. 
* `per_area`- Totall percent area of B.boxes covered per unique image.  
* `max_area` - Largest B.box area covered per unique image. 
* `brightness`- Brightness Percentage.

<img src="https://user-images.githubusercontent.com/64481847/92125206-63dacc80-ee1c-11ea-917d-729e44318034.png" height="300" weidth="1200">

**SOURCE INFO.**:
* **ARVALIS** - Plant Institute. Institut du Végétal: ARVALIS - Institut du vegetal is an applied agricultural research organization dedicated to arable crops : cereals, maize, sorghum, potatoes, fodder crops, flax and tobacco. … It considers technological innovation as a major tool to enable producers and agri-companies to respond to societal challenges.

* **ETHZ**- ETH Zurich trains true experts and prepares its students to carry out their tasks as critical members of their communities, making an important contribution to the sustainable development of science, the economy and society.

* **INRAE** is France's new National Research Institute for Agriculture, Food and Environment, created on January 1, 2020, It was formed by the merger of INRA, the National Institute for Agricultural Research, and IRSTEA, the National Research Institute of Science and Technology for the Environment and Agriculture.

* **RRES** 90003.
Rothamsted Research is the longest-running agricultural research institution in the world. Its foundation dates from 1843 when John Bennet Lawes, the owner of the Rothamsted Estate, appointed Joseph Henry Gilbert, a chemist, as his scientific collaborator and planted the first of what were to become the classical Rothamsted long term experiments, on Broadbalk field. The scientific partnership between Lawes and Gilbert lasted 57 years, and together they laid the foundations of modern scientific agriculture and established the principles of crop nutrition. https://www.rothamsted.ac.uk/history-and-heritage

* University of Saskatchewan (**USask**) researchers played a key role in an international consortium that has sequenced the entire genome of durum wheat—the source of semolina for pasta, a food staple for the world's population, according to an article published today in Nature Genetics.


**TOP data Contributers (Sources)**
* arvalis_1
* ethz_1
* arvalis
* rres_1

<img src="https://user-images.githubusercontent.com/64481847/92126383-bbc60300-ee1d-11ea-9709-1b6a40f88c62.png" height="500" weidth="1500">

<img src="https://user-images.githubusercontent.com/64481847/92127147-a43b4a00-ee1e-11ea-88eb-905c54acd213.png" height="500" weidth="1500">

**Bright image contributers**
* arvalis_1
* arvalis_2
* i.e *They have collected image in Day-Light condition.*

**Dark(dim) image contributers**
* rres_1
* inrae_1
* i.e *They have collected image in Low-Light condition. Also may the area is very Dense or Deep.*

<img src="https://user-images.githubusercontent.com/64481847/92127315-cdf47100-ee1e-11ea-93c1-9c6d76c5d273.png" height="500" weidth="1500">


<img src="https://user-images.githubusercontent.com/64481847/92127791-4b1fe600-ee1f-11ea-8302-5f4d1f977964.png" height="500" weidth="1500">

<img src="https://user-images.githubusercontent.com/64481847/92127876-62f76a00-ee1f-11ea-83ba-0b11275a8237.png" height="500" weidth="1500">


# Image Data Analysis 🎞️:

<img src="https://user-images.githubusercontent.com/64481847/92133605-32ff9500-ee26-11ea-9b70-5341851f28be.jpg"  weidth="1300">

<br>

<img src="https://user-images.githubusercontent.com/64481847/92133619-35fa8580-ee26-11ea-9ece-b6809d40e667.jpg"  weidth="1300">

<br><br><br>

# Outlier fixation 🛠️:
<img src="https://user-images.githubusercontent.com/64481847/92133625-385cdf80-ee26-11ea-8c0b-b1000c198d4e.jpg"  weidth="1300">

### We have encountered with Mis-matched Large Bounding Boxes in certain images in EDA
#### Bounding Boxes Area Distribution( With Outliers)

![out](https://user-images.githubusercontent.com/64481847/92137488-ccc94100-ee2a-11ea-91e2-fa1246074aaa.png)

### Bounding Box Area outliers:
* We can see majority of the Bounding Boxes coverd less than 4% of totall area on Actuall Image.
* Outliers are those BBoxes which have coverd more than 4% of tatall area of Image, as WHEAT HEAD covers less space in an image.
* Also we will discard those Boxes with area less than 0.3%

#### Bounding Boxes Area Distribution( Without Outliers)

![out1](https://user-images.githubusercontent.com/64481847/92137476-c9ce5080-ee2a-11ea-9a0c-19cebbdf29e7.png)



# Data Per-Processing for Keras RetinaNet 👨‍💻:

### Annotations format:
The CSV file with annotations should contain one annotation per line. Images with multiple bounding boxes should use one row per bounding box. Note that indexing for pixel values starts at 0. The expected format of each line is:

`'path/to/image.jpg', x_min, y_min, x_max, y_max, class_name`


![whqd](https://user-images.githubusercontent.com/64481847/92138614-38f87480-ee2c-11ea-8658-56dc36e2cdb2.png)


Also we have to specify the label mapping `classes.csv` as follows:

`with open("classes.csv","w") as file:
    file.write("wheat_head,0")`
    

# Keras RetinaNet Installation🧰:
### Super Thankfull to : **https://github.com/fizyr/keras-retinanet**
* Clone this⏫ repository.
* Ensure numpy is installed using `pip install numpy --user`
* In the repository, execute `pip install . --user`. Note that due to inconsistencies with how tensorflow should be installed, this package does not define a dependency on tensorflow as it will try to install that (which at least on Arch Linux results in an incorrect installation). Please make sure tensorflow is installed as per your systems requirements.
* Alternatively, you can run the code directly from the cloned repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.

### You can Use Pre-trained models using this <a href="https://github.com/Adk2001tech/Global-Wheat-Head-Detection/blob/master/Retinanet_per_trained/Retinanet_pertrained.py">SCRIPT</a>.

# Traning RetinaNet Model ⚙️:
**Notebook link:**
* <a href="https://www.kaggle.com/akhileshdkapse/wheat-head-detection-with-retinanet-keras">**Kaggle Notebook**</a>
* <a href="https://github.com/Adk2001tech/Global-Wheat-Head-Detection/blob/master/Wheat-head-detection-with-retinanet-keras.ipynb">**GitHub Repository Notebook**</a>

1. Download Pre-trained Base models [`resnet101, resnet50, resnet152`]
2. Save it in `snapshots/pretrained_model.h5`.
3. Get ready with `annotations.csv` and `classes.csv` files in same running directory.
4. Then Run the following command:

       !keras_retinanet/bin/train.py --freeze-backbone \
        --random-transform \
        --weights {snapshots/pretrained_model.h5} \
        --batch-size 8 \
        --steps 200 \
        --epochs 10 \
        csv annotations.csv classes.csv
        
5. While traning, model will get saved in each epoch at `snapshots/` folder. With More compu
6. For Loading the Traned model, we can follow this:

            model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
            model = models.load_model(model_path, backbone_name='resnet50')
            model = models.convert_model(model)

7. You can use `perd_from_model(model, image, th=0.5, box_only=False)` from this <a href="https://github.com/Adk2001tech/Global-Wheat-Head-Detection/blob/master/wheat_head_video/Wheat_Head_VIDEO_processing.py">SCRIPT</a> to draw bounding boxes on an Image.

8. Download Trained(8 epochs) Wheat head Detection Model from <a href="https://drive.google.com/file/d/1xUMA2SOOXj4tBw0rg4Bbe2phWfXIVe7L/view?usp=sharing">here</a>. You can go 
ahead for more epoch with more computational💻 power.
# Visualizing model performance 🔖:
## 📸
![pred2](https://user-images.githubusercontent.com/64481847/92199401-af30c180-ee94-11ea-8bc2-a55450d2c35e.png)

-------------------------------------------------------------
![pred3](https://user-images.githubusercontent.com/64481847/92199454-d8e9e880-ee94-11ea-8eca-d3a58f21748a.png)

-------------------------------------------------------------
## 🎦

![wheat](https://user-images.githubusercontent.com/64481847/92202310-52d1a000-ee9c-11ea-8a8e-77b02e845561.gif)


# Python Packages📦 used:

* **Numpy**: NumPy stands for Numerical Python. It support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
* **Pandas**: pandas is a Python package that provides fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python.
* **Matplotlib**: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It can be used in Python scripts, the Python and IPython shell, web application servers, and various graphical user interface toolkits.
* **Plotly**: It is an interactive, open-source, and browser-based graphing library for Python.
* **Seaborn**: Seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures.


<p float="left">
  <img src="https://user-images.githubusercontent.com/64481847/92205366-2a00d900-eea3-11ea-9389-22d5b9911985.jpg" width="400" />
  <img src="https://user-images.githubusercontent.com/64481847/92205324-13f31880-eea3-11ea-8dfc-1710799ad18b.jpg" width="400" />
</p>

<br>

* **Tqdm**: Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you’re done! tqdm works on any platform (Linux, Windows, Mac, FreeBSD, NetBSD, Solaris/SunOS), in any console or in a GUI, and is also friendly with IPython/Jupyter notebooks.
* **OpenCv**: OpenCV is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture and related features analysis. OpenCV runs on the following desktop operating systems: Windows, Linux, macOS, FreeBSD, NetBSD, OpenBSD.
* **Tensorflow**: TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google.
* **Keras**: Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides.

<p float="left">
  <img src="https://user-images.githubusercontent.com/64481847/92205356-20777100-eea3-11ea-961e-f48bcfddb8eb.jpeg" width="400" /> 
  <img src="https://user-images.githubusercontent.com/64481847/92205363-279e7f00-eea3-11ea-8281-c28100d175f4.jpeg" width="400" />
  <img src="https://user-images.githubusercontent.com/64481847/92205361-266d5200-eea3-11ea-9967-a39f80ca61ac.jpeg" width="700" />
</p>

## Thank you for reading :-)
