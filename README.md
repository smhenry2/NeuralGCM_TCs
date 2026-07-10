# NeuralGCM_TCs
Repository containing code for processing and analysis for a NeuralGCM attribution study on TCs

The code to run NeuralGCM is based off the January 21st commit to GitHub. The following lines can be used to reproduce the environment NeuralGCM was run in:
```
pip install git+https://github.com/google-research/neuralgcm.git@5484cf580aa67bc85ad830b8ad80a15b23bc385f
pip install -U "jax[cuda12]"
pip install gcsfs
```

The function below was used to create the perturbations for the ensemble simulations of NeuralGCM 1.4 degree determinisitc model.

```
def perturbation(lat):
        if lat < 60 and lat > (-1 * 60):
            r = (np.random.uniform(0, 1) - 0.5) * np.exp(-(lat**2) / 50**2) / 1000
        else:
            r = 0.0
        return r
```
