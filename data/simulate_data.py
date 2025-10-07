import jax.random as jr
import jax.numpy as jnp
import pandas as pd
import numpy as np
import os
import dill

# Create key
rng = jr.PRNGKey(12)

# Path to save data
outdir = "/Users/melodiemonod/git/neuralsurv/data/data_files"

# Set parameters
num_samples_train_1 = 25
num_samples_train_2 = 50
num_samples_train_3 = 100
num_samples_train_4 = 150
num_samples_test = 100
num_samples_val = 100
rate_censoring = 0.025
dim_x = 4


# Get time-to-event and time-to-censoring distributions
def time_to_event_sample(key, num_samples, x):
    key, key0, key1 = jr.split(key, 3)
    samples_0 = jnp.exp(
        3.0 + 0.8 * jr.normal(key0, shape=(num_samples,))
    )  # log normal(3, 0.8)
    samples_1 = jnp.exp(
        3.5 + 1.0 * jr.normal(key1, shape=(num_samples,))
    )  # log normal(3.5, 1.0)
    return jnp.where(x[:, 0].squeeze() == 0, samples_0, samples_1)


def time_to_censoring_sample(key, num_samples):
    return jr.exponential(key, (num_samples,)) / rate_censoring


def generate_data(rng, num_samples):
    # Simulate covariates (x)
    rng, subrng = jr.split(rng)
    x = jr.normal(subrng, (num_samples, dim_x))
    x = x.at[:, 0].set(x[:, 0] > 0)

    # Simulate time-to-event and time-to-censing
    rng, subrng_tte, subrng_ttc = jr.split(rng, 3)
    time_to_event_obs = time_to_event_sample(subrng_tte, num_samples, x)
    time_to_censoring_obs = time_to_censoring_sample(subrng_ttc, num_samples)

    # Generate time = minimum and event = event or censoring?
    time = jnp.minimum(time_to_event_obs, time_to_censoring_obs)
    event = time_to_event_obs <= time_to_censoring_obs

    # Format data
    num_columns = x.shape[1]
    df_train = pd.DataFrame(x, columns=[f"x_{i}" for i in range(num_columns)])
    df_train["event"] = np.float32(event)
    df_train["time"] = np.float32(time)
    array_train = {
        "x": np.float32(x),
        "time": np.float32(time),
        "event": np.float32(event),
    }
    return {"pd.DataFrame": df_train, "np.array": array_train}


# Generate train data
rng, rng_train_1, rng_train_2, rng_train_3, rng_train_4, rng_test, rng_val = jr.split(
    rng, 7
)
train_data_1 = generate_data(rng_train_1, num_samples_train_1)
train_data_2 = generate_data(rng_train_2, num_samples_train_2)
train_data_3 = generate_data(rng_train_3, num_samples_train_3)
train_data_4 = generate_data(rng_train_4, num_samples_train_4)
test_data = generate_data(rng_test, num_samples_test)
val_data = generate_data(rng_val, num_samples_val)

while train_data_1["np.array"]["time"].max() < test_data["np.array"]["time"].max():
    rng, rng_train_1 = jr.split(rng, 2)  # avoid max time test > max time train for ibs
    train_data_1 = generate_data(rng_train_1, num_samples_train_1)

# Save
synthetic_data_1_path = os.path.join(outdir, "synthetic_data_1.pkl")
synthetic_data_1 = {"train": train_data_1, "test": test_data, "val": val_data}
with open(synthetic_data_1_path, "wb") as f:
    dill.dump(synthetic_data_1, f)

synthetic_data_2_path = os.path.join(outdir, "synthetic_data_2.pkl")
synthetic_data_2 = {"train": train_data_2, "test": test_data, "val": val_data}
with open(synthetic_data_2_path, "wb") as f:
    dill.dump(synthetic_data_2, f)

synthetic_data_3_path = os.path.join(outdir, "synthetic_data_3.pkl")
synthetic_data_3 = {"train": train_data_3, "test": test_data, "val": val_data}
with open(synthetic_data_3_path, "wb") as f:
    dill.dump(synthetic_data_3, f)

synthetic_data_4_path = os.path.join(outdir, "synthetic_data_4.pkl")
synthetic_data_4 = {"train": train_data_4, "test": test_data, "val": val_data}
with open(synthetic_data_4_path, "wb") as f:
    dill.dump(synthetic_data_4, f)
