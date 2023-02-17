import pywt
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    print(pywt.wavelist())
    out_dir = Path("E:/thesis_final_feb16/wavelet")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["morl", "mexh"]:
        [psi, x] = pywt.ContinuousWavelet(name).wavefun()
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot(x, psi)
        ax.set_ylabel(f'Amplitude')
        ax.set_xlabel('Time')
        ax.set_title(f'{name} Wavelet'.title())
        ax.grid()
        fig.tight_layout()
        filename = f"{name}_wavelet.png"
        filepath = out_dir / filename
        print(filepath)
        fig.savefig(filepath)
        plt.show()