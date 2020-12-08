
# Neural Unsigned Distance Fields
> Neural Unsigned Distance Fields for Implicit Function Learning <br />
> [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html), [Aymen Mir](http://virtualhumans.mpi-inf.mpg.de/people/Mir.html), [Gerard Pons-Moll](http://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)

![Teaser](ndf-teaser.png)

[Paper](http://virtualhumans.mpi-inf.mpg.de/papers/chibane2020ndf/chibane2020ndf.pdf) - 
[Supplementaty](http://virtualhumans.mpi-inf.mpg.de/papers/chibane2020ndf/chibane2020ndf-supp.pdf) -
[Project Website](http://virtualhumans.mpi-inf.mpg.de/ndf/) -
[Arxiv](https://arxiv.org/abs/2010.13938) -
Video -
Published in NeurIPS 2020.


#### Citation
If you find our project useful, please cite the following.

    @inproceedings{chibane2020ndf,
        title = {Neural Unsigned Distance Fields for Implicit Function Learning},
        author = {Chibane, Julian and Mir, Aymen and Pons-Moll, Gerard},
        booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
        month = {December},
        year = {2020},
    }

## Install

A linux system with cuda 10 is required for the project.

The `NDF_env.yml` file contains all necessary python dependencies for the project.
To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:
```
conda env create -f NDF_env.yml
conda activate NDF
```

Please clone the repository and navigate into it in your terminal, its location is assumed for all subsequent commands.

The repository currenlty holds the data for the experiment on full (not closed) ShapeNet Car class with 10.000 input points. Further experimental setups will soon follow. 

## Using Pretrained Model
Please download the needed data from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/Nc6qWEfseH7J7Sz), and unzip it into `shapenet\data` - unzipped files require 150 GB free space. 

Next, you can start generation of instances from the test set via
```
python generate.py -pretrained
```


## Training and generation
To train NDF yourself use
```
python train.py 
```


In the `experiments/` folder you can find an experiment folder containing the model checkpoints, the checkpoint of validation minimum, and a folder containing a tensorboard summary, which can be started at with
```
tensorboard --logdir experiments/YOUR_EXPERIMENT/summary/ --host 0.0.0.0
```

To generate results for instances of the test set, please use
```
python generate.py
```


## Note & Contact

**Further experiments and updates will follow shortly.**
For questions and comments regarding the code please contact [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html) via mail. (See Paper)

## License
Copyright (c) 2020 Julian Chibane, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.
For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the `Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion` paper in documents and papers that report on research using this Software.
