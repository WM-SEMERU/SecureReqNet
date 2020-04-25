# SecureReqNet

>
> Maintained by @danaderp. Last update: April 2020
>

We present a machine learning approach, named *SecureReqNet*, to automatically identify whether issues describe security related content. *SecureReqNet* hinges on the idea of predicting severity on software using vulnerability desciptions [(Han, et al., 2017)](https://ieeexplore.ieee.org/abstract/document/8094415) by incorporating desing principles from AlexNet (Krizhevsky, et al., 2013). 

(place image here)

*SecureReqNet* consists of a two-phase deep learning architecture that operates *(for now)* purely on the natural language descriptions of issues. The first phase of our approach learns high dimensional sentence embeddings from hundreds of thousands of descriptions extracted from software vulnerabilities listed in the CVE database and issue descriptions extracted from open source projects using an unsupervised learning process. The second phase then utilizes this semantic ontology of embeddings to train a deep convolutional neural network capable of predicting whether a given issue contains security-related information.


*SecureReqNet* has four versions that vary in terms of the size of the tensors and the parameters of the convolutional layers.

1. **SecureReqNet (shallow)** was based on the best architecture achived by Han, et al. Such architecture implemented one convolution layer with 3 kernes of different sizes. The authors set up the size of each kernel as 1-gram, 3-gram, and 5-gram to reduce an input matrix. This matrix was built by means of an unsupervised word2vec where the rows represents the words in a given document (or issue) and the columns the size of the embedding. Details of how we trained our word2vec can be found in the notebook [*03_Clustering*](https://github.com/danaderp/SecureReqNet/blob/master/nbs/03_Clustering.ipynb).  **SecureReqNet (shallow)** has a max pooling layer followed by a flatten function. The final tensor is a merged vector from the 3 initial kernels. Unlike Han, et al.' SVM multi-class output layer, we utilized a binary classification throughout a softmax layer. 

2. **SecureReqNet (deep)** was an expansion of **SecureReqNet (shallow)**. We included an extra convolutional layer, a max pooling, and a flatten function. The final tensor is a merged vector from the 3 initial kernels. A fully connected sigmoid layers was added just before the binary softmax layer. 

```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

3. **Alex-SecureReqNet (deep)** was based on the proposed architecture by Krizhevsky et al., where 5 convolutional layers extract the abstract features and 3 fully connected reduce the dimensionality. This is the classical convolutional ImageNet network with a small adaptation in the final layer to induce binary classification. 

4. **α-SecureReqNet (deep)** was a modification of the **Alex-SecureReqNet (deep)** in the convolutional layers. The modification consisted in implementing the n-gram kernel strategy for text-based datasets [(Han, et al., 2017)](https://ieeexplore.ieee.org/abstract/document/8094415). The input layer is a document embedding in the shape of a matrix. The first convolutional layer has a kernel of 7-gram size to reduce the input matrix into 32 vector feature maps. Later, it is applied a max pooling and a flatten function to obtain a column matrix. The second convolutional layer has a 5-gram filter followed by a max pooling and flatten function that merged 64 features. The third, fourth, and fifth convolutional layers are very similar to the original distribution in ImageNet but using 3-gram filters and 128/64 features respectively. Three fully connected layers went after the fifth conv layer to reduce the dimensionality and control the overfitting with the dropout units. The final layer is again a binary softmax layer (security vs non-security related). 

## Datasets

## Research and Components Roadmap
- [x] Using Shallow Neural Network to predict security relatedness on issues (or requirements) 
- [x] Using Deep Neural Network to predict security relatedness on issues (or requirements)
- [ ] Using a Neural Network to predict quantitatively (regression) how critical is an issue (or requirement)
- [ ] Implementing a Transformer Architecture to predict security criticality on issues (or requirements)
- [ ] Recovering security related relationships among software artifacts by employing traceability theory
