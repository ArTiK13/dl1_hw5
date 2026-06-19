from torch import nn

from src.lensless.admm import ADMMReconstructor
from src.lensless.fista import FISTAReconstructor
from src.model.drunet import make_drunet
from src.utils.io_utils import keep_roi


class LenslessReconstructionModel(nn.Module):
    def __init__(
        self,
        method="admm100",
        admm_iterations=100,
        trainable_admm=False,
        preprocessor=None,
        postprocessor=None,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        fista_iterations=100,
    ):
        super().__init__()
        self.method = method
        self.preprocessor = make_drunet(preprocessor) if preprocessor is not None else None
        self.postprocessor = make_drunet(postprocessor) if postprocessor is not None else None
        if method == "fista":
            self.reconstructor = FISTAReconstructor(num_iterations=fista_iterations)
        else:
            self.reconstructor = ADMMReconstructor(
                iterations=admm_iterations,
                trainable=trainable_admm,
                mu1=mu1,
                mu2=mu2,
                mu3=mu3,
                tau=tau,
            )

    def forward(self, measurement=None, psf=None, **batch):
        x = measurement
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        admm_output = self.reconstructor(x, psf)
        prediction = admm_output
        if self.postprocessor is not None:
            prediction = self.postprocessor(prediction)
        prediction = keep_roi(prediction)
        result = {"prediction": prediction}
        if self.postprocessor is not None:
            result["admm_output"] = admm_output
        return result
