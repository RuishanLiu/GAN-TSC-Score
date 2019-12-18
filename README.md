# Compression Score

**Related link: [Codes for GAN-TSC](https://github.com/RuishanLiu/GAN-TSC)**

Codes to compute Compression Score in Paper [Teacher-Student Compression with Generative Adversarial Networks](https://arxiv.org/pdf/1812.02271.pdf).

* To compute Compression Score for a given Cifar10 dataset (real, GAN, noisy, etc), download both compression_score.py and compression_teacher.npy. 

```bash
from compression_score import get_compression_score
data = # data in shape (50000, 32, 32, 3)
# Compute the mean and variance of Compression Score
mean, var = get_compression_score(data, path_teacher='compression_teacher.npy')
```

See example.py as an example usage.
