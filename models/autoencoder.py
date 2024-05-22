import torch

from models.quantize import VARQuantizer
from dac import DAC


class MultiScaleDAC(DAC):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 8, 4, 2],
        codebook_dim: int = 16,
        n_scales: int = 16,
        sample_rate: int = 44100,
    ):
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        super().__init__(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            codebook_dim=codebook_dim,
            sample_rate=sample_rate
        )

        self.quantizer = VARQuantizer(
            input_dim=latent_dim,
            codebook_dim=codebook_dim,
            n_scales=n_scales,
        )

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, aux_loss = self.quantizer(z)
        return z, codes, aux_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "aux_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, aux_loss = self.encode(audio_data)

        x = self.decode(z)

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "aux_loss": aux_loss,
        }


if __name__ == "__main__":
    import dac
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    multiscale = MultiScaleDAC().to("cuda")

    multiscale.encoder.load_state_dict(model.encoder.state_dict())
    multiscale.decoder.load_state_dict(model.decoder.state_dict())

    print(multiscale)





