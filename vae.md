# VAE

## Research Focus

-   representation learning
-   anomaly detection

## Introductory

-   <https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html> overview
-   <https://ryanloweift6266.wordpress.com/2016/02/28/variational-autoencoders/>
-   <https://anotherdatum.com/vae.html> <https://anotherdatum.com/vae2.html>
-   <https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/>
-   <https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/> on CVAE
-   <https://blog.keras.io/building-autoencoders-in-keras.html>
-   <http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/>
-   <http://pyro.ai/examples/vae.html>
-   <https://www.jeremyjordan.me/variational-autoencoders/>
-   <https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf>
-   <https://towardsdatascience.com/a-wizards-guide-to-adversarial-autoencoders-part-1-autoencoder-d9a5f8795af4?> adversarial VAE
-   <https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained> on KL divergence
-   <https://gabrielhuang.gitbooks.io/machine-learning/content/reparametrization-trick.html> on reparametrization trick

## Curated Lists

-   <https://github.com/debasishg/ml-readings/blob/master/vae.md>
-   <https://github.com/matthewvowels1/Awesome-VAEs>


## Models

 | model | paper | implementation |
| ----- | ----- | ---- |
| DAGMM     | AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION     | <https://github.com/danieltan07/dagmmDEEP>     |
| GRU-VAE  | Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network | <https://github.com/smallcowbaby/OmniAnomalyRobust> |
| GMM-GRU-VAE | Time Series Anomaly Detection: A GRU-based Gaussian Mixture Variational Autoencoder Approach  | <https://github.com/jariasf/GMVAE>  |
| GMM-GRU-VAE | DEEP UNSUPERVISED CLUSTERING WITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS | <https://github.com/psanch21/VAE-GMVAEMultidimensional> |
| LSTM-VAE | Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder | <https://github.com/Danyleb/Variational-Lstm-Autoencoder> |
| Adversarial VAE | Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection | <https://github.com/YeongHyeon/adVAESelf-adversarial> |
| Adversarial VAE | learned one-class classifier for novelty detection | <https://github.com/khalooei/ALOCC-CVPR2018Adversarially> |
| Adversarial VAE | Probabilistic Novelty Detection with Adversarial Autoencoders | <https://github.com/podgorskiy/GPNDGenerative> |
| GAN+VAE | Disentangling factors of variation in deep representations using adversarial training | <https://github.com/MichaelMathieu/factors-variation> <https://github.com/ananyahjha93/disentangling-factors-of-variation-using-adversarial-training> |
| HFVAE v1  | Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data | <https://github.com/wnhsu/FactorizedHierarchicalVAE> |
| HFVAE v2 | Scalable Factorized Hierarchical Variational Autoencoder Training | <https://github.com/wnhsu/ScalableFHVAE> |
| HSI-contraint VAE | Information Constraints on Auto-Encoding Variational Bayes | <https://github.com/romain-lopez/HCV> |
| CEVAE | Effect Inference with Deep Latent-Variable Models | <https://github.com/Emr03/CEVAECausal> <https://github.com/AMLab-Amsterdam/CEVAE> |
| β-VAE | β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK | <https://github.com/google-research/disentanglement_lib> <https://github.com/cianeastwood/qedr> |
| FactorVAE | Disentangling by Factorising | <https://github.com/google-research/disentanglement_lib> <https://github.com/1Konny/FactorVAE> <https://github.com/AliLotfi92/Disentangling_by_Factorising> <https://github.com/paruby/FactorVAE> <https://github.com/nicolasigor/FactorVAE> |
| BetaTCVAE | Isolating Sources of Disentanglement in VAEs | <https://github.com/google-research/disentanglement_lib> <https://github.com/rtqichen/beta-tcvae> |
| DIP-VAE | VARIATIONAL INFERENCE OF DISENTANGLED LATENT CONCEPTS FROM UNLABELED OBSERVATIONS |<https://github.com/paruby/DIP-VAE> <https://github.com/IBM/AIX360/blob/master/aix360/algorithms/dipvae/dipvae.py><br/><https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py>  <https://github.com/ethanluoyc/pytorch-vae> |
| InfoVAE | InfoVAE: Balancing Learning and Inference in Variational Autoencoders |<https://github.com/ShengjiaZhao/InfoVAE> |
| WAE |Wasserstein Auto-Encoders |<https://github.com/tolstikhin/wae> |
| LVAE | Ladder Variational Autoencoders | <https://github.com/ermongroup/Variational-Ladder-Autoencoder> <https://github.com/Michedev/VLAE> <https://github.com/davidsandberg/LadderVAE> <https://github.com/addtt/ladder-vae-pytorch> |
| VFAE |THE VARIATIONAL FAIR AUTOENCODER | <https://github.com/dendisuhubdy/vfae> <https://github.com/yevgeni-integrate-ai/VFAE> |
| SVAE |Composing graphical models with neural networks for structured representations and fast inference | <https://github.com/mattjj/svae> |
| VQ-VAE | Neural Discrete Representation Learning | <https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py> <https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb> <https://github.com/hiwonjoon/tf-vqvae> <https://github.com/1Konny/VQ-VAE> <https://github.com/ritheshkumar95/pytorch-vqvae> <https://github.com/swasun/VQ-VAE-Speech> <https://github.com/DongyaoZhu/VQ-VAE-WaveNet> <https://github.com/swasun/VQ-VAE-Speech> <https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation> <https://github.com/MishaLaskin/vqvae> <https://github.com/andrecianflone/vector_quantization> |
| VQ-VAE-2 | Generating Diverse High-Fidelity Images with VQ-VAE-2 | <https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb> <https://github.com/rosinality/vq-vae-2-pytorch> <https://github.com/unixpickle/vq-vae-2> |
| DDVAE |Learning to Decompose and Disentangle Representations for Video Prediction |<https://github.com/jthsieh/DDPAE-video-prediction> |
|  SOM-VAE |SOM-VAE: INTERPRETABLE DISCRETE REPRESENTATION LEARNING ON TIME SERIES |<https://github.com/KurochkinAlexey/SOM-VAE> <https://github.com/ratschlab/SOM-VAE> |
| DRAW |DRAW: A Recurrent Neural Network For Image Generation |<https://github.com/ericjang/draw> <https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW> <https://github.com/jbornschein/draw> <https://github.com/vivanov879/draw> <https://github.com/chenzhaomin123/draw_pytorch> <https://github.com/suhoy901/DRAW_pytorch> <https://github.com/conan7882/DRAW-recurrent-image-generation> <https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW> |
| CVAE |Learning Structured Output Representation using Deep Conditional Generative Models |<https://github.com/wsjeon/ConditionalVariationalAutoencoder> <https://github.com/zafarali/generative> |
| JointVAE | Learning Disentangled Joint Continuous and Discrete Representations | <https://github.com/Schlumberger/joint-vae>  <https://github.com/voxmenthe/JointVAE_v1> |
| TD-VAE |  TEMPORAL DIFFERENCE VARIATIONAL AUTO-ENCODER | <https://github.com/xqding/TD-VAE> <https://github.com/ankitkv/TD-VAE> <https://github.com/MillionIntegrals/td-vae> |
