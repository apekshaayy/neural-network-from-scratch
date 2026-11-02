"""
Streamlit app comes here.
"""

"""
Streamlit app comes here.
"""


import streamlit as st
import pandas as pd
import torch

from mnist import Net, train, train_loader, MEAN, STD
from corruptions import CORRUPTIONS, load_shifted
from eval import make_norm, make_tent


CHECKPOINT = "mnist_bn.pt"

st.set_page_config(
    page_title="MNIST Test-Time Adaptation",
    page_icon=None,
    layout="wide"
)


@st.cache_resource
def load_source_model():
    """Load the trained model, or train it once if no checkpoint exists."""
    model = Net()

    if not Path(CHECKPOINT).exists():
        with st.spinner("Training the source model for the first time..."):
            train(model, train_loader, epochs=3)
            torch.save(model.state_dict(), CHECKPOINT)
    else:
        model.load_state_dict(
            torch.load(CHECKPOINT, map_location="cpu")
        )

    model.eval()
    return model


@st.cache_data
def get_shifted_data(corruption):
    """Load raw corrupted images and their labels."""
    return load_shifted(corruption, normalize=False)


def evaluate(model, images, labels, batch_size=128):
    """Evaluate a model without using the labels for adaptation."""
    correct = 0

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        targets = labels[start:start + batch_size]

        with torch.no_grad():
            predictions = model(batch).argmax(dim=1)

        correct += (predictions == targets).sum().item()

    return correct / len(images)


st.title("MNIST Test-Time Adaptation")
st.write(
    "Compare a trained source model with test-time normalization "
    "and Tent under different image corruptions."
)

st.sidebar.header("Experiment settings")

corruption_names = list(CORRUPTIONS.keys())
corruption = st.sidebar.selectbox(
    "Select image corruption",
    corruption_names,
    index=1
)

num_samples = st.sidebar.slider(
    "Number of test images",
    min_value=500,
    max_value=10000,
    value=2000,
    step=500
)

learning_rate = st.sidebar.select_slider(
    "TENT learning rate",
    options=[0.0001, 0.0005, 0.001, 0.005],
    value=0.001,
    format_func=lambda value: f"{value:g}"
)

st.sidebar.caption(
    "The same test images are used for all three methods."
)

if st.button("Run comparison", type="primary"):
    source = load_source_model()

    with st.spinner("Loading and preparing the test data..."):
        raw_images, labels = get_shifted_data(corruption)

        images = (raw_images - MEAN) / STD
        images = images[:num_samples]
        labels = labels[:num_samples]

    with st.spinner("Evaluating the source model..."):
        source_accuracy = evaluate(source, images, labels)

    with st.spinner("Evaluating test-time normalization..."):
        norm_model = make_norm(source)
        norm_accuracy = evaluate(norm_model, images, labels)

    with st.spinner("Adapting and evaluating TENT..."):
        tent_model = make_tent(source, lr=learning_rate)
        tent_accuracy = evaluate(tent_model, images, labels)

    results = pd.DataFrame({
        "Method": ["Source", "Norm", "TENT"],
        "Accuracy": [
            source_accuracy * 100,
            norm_accuracy * 100,
            tent_accuracy * 100
        ]
    })

    st.subheader("Accuracy comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("Source Model", f"{source_accuracy:.2%}")
    col2.metric("Norm", f"{norm_accuracy:.2%}")
    col3.metric("TENT", f"{tent_accuracy:.2%}")

    st.bar_chart(
        results.set_index("Method"),
        y="Accuracy",
        y_label="Accuracy (%)"
    )

    st.subheader("Corrupted MNIST samples")
    st.caption(f"Corruption: {corruption} · Showing the first 8 images")

    sample_columns = st.columns(8)
    for index, column in enumerate(sample_columns):
        with column:
            st.image(
                raw_images[index, 0].numpy(),
                caption=f"Label: {int(labels[index])}",
                width=75
            )

    st.success("Comparison completed.")

else:
    st.info("Choose an image corruption and click Run comparison.")

    st.subheader("Methods")
    st.write(
        "- **Source:** Uses the original model without test-time adaptation."
    )
    st.write(
        "- **Norm:** Uses batch statistics during testing."
    )
    st.write(
        "- **TENT:** Minimizes prediction entropy by updating "
        "the affine parameters of Batch Normalization layers."
    )
