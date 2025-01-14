# Team 2 Project
#### Members: 
Karolina Naranjo-Velasco and Joseph Choi

## Topic: 
A Survey of Secure and Efficient Medical Image Encryption using Deep Learning Methods

### Motivation:

The promising performance of deep learning methods for the medical application have resulted in more than thousands of communication over the network (e.g., sharing data for a collaborative effort to achieve a richer data set, sharing promising models to each other on the Internet of Medical Things. 

Despite the promising performance of deep learning methods in the application of medical fields, the privacy of the patients’ sensitive information is at high risk as the data becomes increasingly demanding. Even though there are regulations/standards for the privacy of patient’s information, such as the Health Insurance Portability and Accountability Act (HIPAA) and the recent AI bill of rights, the privacy of patients’ information is still at risk. 


### **Deep learning to encrypt medical image data:** 

The Advanced Encryption Standards (AES) is a U.S. federal standard for encryption algorithms for patient information. The AES has an outstanding performance in the speed of encryption/decryption and security. However, its performance can be degraded significantly by knowing the general pattern of how the private key has been generated. The encryption algorithms with known forms and the process of algorithms allow an attacker to hack the system. 

The performance of the deep learning is noted in many different applications. Despite DL's prediction performance, DL is often discriminated for being a "black box" model as the interpretation of the prediction is hardly achivable. However, being a "black box" model could actually be a benefit for some applications, especially for the image encryption. Because nobody knows how the actual encrpytion is done using the DL methods, DL based image encryption creates a lot of potentials. We limited our scope to the deep learning based image encryption to the field of medical imagings (e.g., X-ray, MRI, etc.). There are few survey papers discussion the deep learning based image encryption method, but not with the deep learning based image encryption method for the medical imaging to the best of our knowledge. 

There could be many ways to encrypt medical images using deep learning. The application of the deep learning for medical image encryption is a very recent trends, and thus, our survey focuses on the papers mostly published after 2018. 

We observed the application of the deep learning methods fall under some general categories of the methods of the following: 

* GAN based methods
* Autoencoders, Variational Autoencoders
* invertible neural networks
* forward neural networks 

We would love to compare those methods in the benchmark dataset (e.g., chest X-ray, brain MRI, etc.) for the benchmark metrics (e.g., loss of information from decription, etc.), but most of the papers in scope **do not** have their implementation available for us. 

However, most of the papers validated their methods for some common benchmark data set and benchmark metrics. Thus, we would utilize their reported result and create a comprehensive table summarizing and comparing different methods. 
