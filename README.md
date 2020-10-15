# Teacher-Student Compression Score (TSCS)

* To compute Compression Score for a given Cifar10 dataset (real, GAN, noisy, etc), download both compression_score.py and compression_teacher.npy. 

```bash
from compression_score import get_compression_score
data = # data in shape (50000, 32, 32, 3)
# Compute the mean and variance of Compression Score
mean, var = get_compression_score(data, path_teacher='compression_teacher.npy')
```

See example.py as an example usage.
