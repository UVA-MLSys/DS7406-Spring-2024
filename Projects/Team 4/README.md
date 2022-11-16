# Team 4 Project
Members: Navya Annapareddy and Jade Preston
Topic: A Survey of Federated Learness, Privacy, and Fairness in Healthcare

**Problem Description:** Currently, there are many contexts where data sharing is difficult or constrained by security and distribution limitations. One common domain where this is a consideration is in Healthcare where data is often governed by data-use-ordinances like HIPAA. On the other hand, larger sample sizes allow models to better generalize on account of the potential for more variability and balancing underrepresented classes. 
Federated learning is a type of distributed learning model that allows data to be trained in a decentralized manner. This, in turn, addresses data security, privacy, and vulnerability considerations as data itself is not shared across a given learning network’s nodes. Some challenges to federated learning include: node data may not be independent and identically distributed (iid), relatively high levels of communication between network machines is needed, and heterogeneity in the individual nodes with respect to bias and size of data samples. 

![Taxonomy](https://github.com/UVA-MLSys/DS7406/blob/main/Projects/Team%205/images/taxonomy.png)
Previously proposed fairness mechanism taxonomy (Shi et. al)

**Considerations:** Federated learning has much potential to curb security vulnerabilities normally present in data sharing processes. Data protection would still need to be enforced at the host level but would not require a dedicated TEE to ensemble or share models unless desired. Distributed training methods typically focus on parallelization to obtain less computationally expensive training while federated methods have focused on addressing node heterogeneity. Local systems are considered suitable, but  HPC systems might be necessary for high dimensional data. Encryption can be enforced at the model sharing level through secure communication protocols or with data keys (envelope encryption). Finally, secure addition of peers to a network must also be considered.

**Citations:**

Shi, Yuxin, Han Yu, and Cyril Leung. "A survey of fairness-aware federated learning." arXiv preprint arXiv:2111.01872 (2021). 
