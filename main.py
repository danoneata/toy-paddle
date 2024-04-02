import pdb
import random
import time

from collections import Counter

from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from paddle import Paddle

import numpy as np
import seaborn as sns
import streamlit as st
import torch


st.set_page_config(layout="wide")


with st.sidebar:
    shot = st.number_input("num. shots", 1, 50, 15)
    n_classes = st.number_input("num. classes", 2, 50, 5)
    cluster_std = st.number_input("cluster std.", 1.0, 5.0, 1.5, step=0.5)
    n_classes_query = st.number_input("num. query classes", 1, n_classes, 1)
    n_samples_per_class_query = st.number_input(
        "num. query samples per class", 1, 100, 5
    )
    random_state = st.number_input("random seed", 0, 100, 42)
    to_show_intermediate = st.checkbox("show intermediate steps", value=False)

random.seed(random_state)
n_samples = shot * n_classes

X_tr, y_tr, centers = make_blobs(
    n_samples=n_samples,
    n_features=2,
    centers=n_classes,
    # center_box=(-4, 4),
    cluster_std=cluster_std,
    random_state=random_state,
    shuffle=False,
    return_centers=True,
)

idxs = random.sample(range(n_classes), n_classes_query)
centers1 = centers[idxs]
n_samples_query = n_samples_per_class_query * n_classes_query
X_te, _ = make_blobs(
    n_samples=n_samples_query,
    n_features=2,
    centers=centers1,
    # center_box=(-4, 4),
    cluster_std=cluster_std,
    random_state=random_state,
    shuffle=False,
    return_centers=False,
)
y_te = [idxs[i] for i in range(n_classes_query) for _ in range(n_samples_per_class_query)]
y_te = np.array(y_te)

cols = st.columns(2)
cols[0].header("Nearest class mean")
cols[1].header("Paddle")

centers_computed = np.zeros((n_classes, 2))
for c in range(n_classes):
    centers_computed[c] = np.mean(X_tr[y_tr == c], axis=0)



def make_plot(ax, centers):
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="tab10", label="support samples")
    ax.scatter(
        centers_computed[:, 0],
        centers_computed[:, 1],
        marker="*",
        c=list(range(n_classes)),
        cmap="tab10",
        label="centers",
    )
    ax.scatter(X_te[:, 0], X_te[:, 1], c="k", marker=".", label="query samples")

    for c in range(n_classes):
        ax.text(centers[c, 0], centers[c, 1], f"{c}", fontsize=12)

    dist = cdist(X_te, centers)
    closest_centers = np.argmin(dist, axis=1)

    for i, c in enumerate(closest_centers):
        ax.plot(
            [X_te[i, 0], centers[c, 0]],
            [X_te[i, 1], centers[c, 1]],
            c="k",
            alpha=0.5,
        )

    ax.legend()

fig, ax = plt.subplots()
make_plot(ax, centers_computed)
cols[0].pyplot(fig)

paddle_kwargs = {
    "device": "cuda:0",
    "log_file": "log",
    "args": {"iter": 20, "alpha": 1},
}
paddle = Paddle(**paddle_kwargs)


task_dic = {
    "x_s": X_tr.reshape(1, n_classes * shot, 2),
    "y_s": y_tr.reshape(1, n_classes * shot, 1),
    "x_q": X_te.reshape(1, n_classes_query * n_samples_per_class_query, 2),
    "y_q": y_te.reshape(1, n_classes_query * n_samples_per_class_query, 1),
}
# st.write(task_dic["y_s"])
# st.write(task_dic["y_q"])
task_dic = {k: torch.tensor(v).float() for k, v in task_dic.items()}

for u, w in paddle.run_task(task_dic):
    w = w.cpu().numpy()[0]
    try:
        u = u.cpu().numpy()[0]
    except:
        u = None

    if to_show_intermediate:
        fig, ax = plt.subplots()
        make_plot(ax, w)
        ax.scatter(
            w[:, 0],
            w[:, 1],
            marker="x",
            c=list(range(n_classes)),
            cmap="tab10",
            label="centers paddle",
        )
        cols[1].pyplot(fig)
        cols[1].markdown("U")
        cols[1].write(u)

fig, ax = plt.subplots()
make_plot(ax, w)
ax.scatter(
    w[:, 0],
    w[:, 1],
    marker="x",
    c=list(range(n_classes)),
    cmap="tab10",
    label="centers paddle",
)
cols[1].pyplot(fig)
cols[1].markdown("U")
cols[1].write(u)
