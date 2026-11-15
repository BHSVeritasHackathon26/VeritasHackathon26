"""
model_adapter.py  --  THE ONLY FILE YOU NEED TO EDIT to connect the trained model.

The server calls two functions:

    load_model()                    -> called ONCE when the server starts
    predict(model, features, info)  -> called once per trial in the dropped HDF5 file

Until load_model() returns something other than None, the app runs in DEMO MODE
(it shows placeholder text, clearly labelled, so the UI can be tried without a model).

If you trained with the Brain-to-Text '25 baseline code (GRU decoder), see
examples/baseline_gru_adapter.py for a starting point you can paste in here.
"""

# Shown in the app's status pill, e.g. "GRU + 5-gram LM".
MODEL_NAME = "Thought-to-Text model"


def load_model():
    """Load the trained model into memory and return it.

    Return anything you like (a torch module, a dict with model + language
    model + tokenizer, ...). Whatever you return is passed back to predict().

    Example:
        import torch
        model = MyModel(...)
        model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
        model.eval()
        return model
    """
    return None  # None = demo mode. Replace with your real loading code.


def predict(model, features, info):
    """Turn one trial of neural data into a sentence.

    model     whatever load_model() returned
    features  numpy float32 array, shape (time_steps, n_features)
              -- exactly the trial's `input_features` dataset (512 features
              per 20 ms bin in the Brain-to-Text '25 data). Apply the same
              preprocessing here that you used in training.
    info      dict: {"key", "session", "block_num", "trial_num", "filename"}
              (values are None when the file doesn't have them). "session"
              is what you need if your model has per-day input layers.

    Inference only: do not call an optimizer or loss here. The server already
    runs this inside torch.inference_mode(), and load_model() should call
    model.eval() so dropout is off.

    Return the predicted sentence as a string.
    Optionally return a dict instead: {"text": "...", "phonemes": "..."}.
    """
    raise NotImplementedError("Fill in predict() in model_adapter.py")
