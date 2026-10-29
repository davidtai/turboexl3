# TurboEXL3 media kit

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="turboexl3-lockup-animated-dark.svg">
    <img alt="TurboEXL3" src="turboexl3-lockup-animated-light.svg" width="440">
  </picture>
</p>

The TurboEXL3 logo and the files to use it. The mark is the 4x4 Sylvester Hadamard matrix (EXL3 rotates weights with
Hadamard transforms before the trellis codes them): its +1 cells are ink, its -1 cells are faint, and the +1 cell at the
bottom right is blue. The wordmark reads TurboEXL3.

Every logo file comes in two versions: `light` for light grounds (dark ink) and `dark` for dark grounds (light ink).

## Files

| File | What it is |
|---|---|
| `turboexl3-lockup-{light,dark}.svg` | The logo: the mark and the wordmark. |
| `turboexl3-mark-{light,dark}.svg` | The mark alone, for square and small spaces. |
| `turboexl3-lockup-animated-{light,dark}.svg`, `turboexl3-mark-animated-{light,dark}.svg` | The logo building itself: the matrix grows from one cell to 2x2 to 4x4, each copy sliding out of its source and the bottom-right copy flipping sign; then the blue cell pulses and the wordmark fades in. It runs once (2.2 s) and holds. A viewer who asks for reduced motion sees the finished logo. For READMEs and web pages. |
| `png/turboexl3-lockup-{light,dark}@{1x,2x,4x}.png` | The logo on a transparent ground, 408 x 92 px at 1x. |
| `png/turboexl3-mark-{light,dark}-{32,64,128,256,512,1024}.png` | The mark on a transparent ground. The 32 px file is drawn on the pixel grid. |
| `png/turboexl3-avatar-{light,dark}-1024.png` | The mark on a solid ground (white, or navy `#101823`), inside a round crop's safe area: for profile pictures. |
| `gif/turboexl3-loop-{light,dark}.gif` | A 5 s loop that starts and ends on the finished logo: it folds into one cell and rebuilds. 960 x 300 px. |
| `gif/turboexl3-build-{light,dark}.gif` | The build from nothing, once. 960 x 300 px. |
| `social/turboexl3-x-header-{light,dark}.png` | 2000 x 800 (5:2): an X Article header. |
| `social/turboexl3-social-preview-{light,dark}.png` | 1280 x 640: GitHub's social preview (the repository's Settings, Social preview). |
| `video/turboexl3-infographic.mp4` | The results as a 60 s animated infographic: 1920 x 1080, 30 fps, H.264 (BT.709). See Video below. |
| `video/turboexl3-infographic-{1280,960}.gif` | The same video as GIFs, 1280 x 720 and 960 x 540 at 15 fps, looping: for places that play GIFs inline but not video. |
| `video/turboexl3-infographic-poster.png` | Its first frame, 1920 x 1080: a poster image for a video player. |

## Video

The infographic opens on the logo, whole in its first frame, then the tile folds into one cell and rebuilds. It then
walks through the results: accuracy per bit (turboderp's KL divergence against bits per weight); prefill at 16K tokens
and the time to first token, first with Sushi's exact EXL3 kernels and then with turboexl3's; decode by prompt length at
the same MTP depth as the standard setup; coding accuracy against size on disk, with peak memory; and the K4 and K3
trades. It was measured on an Apple M5 Max (128 GB) with mlx-serve and Qwen3.8-Flash-Next; the numbers and the method are
in [RESULTS.md](../RESULTS.md).

## Colours

| | Light version | Dark version |
|---|---|---|
| Ink: the +1 cells and the wordmark | `#16181d` | `#eef1f5` |
| The -1 cells | `#d3d8df` | `#3a4250` |
| The blue cell | `#2a78d6` | `#2a78d6` |

The blue is K5's colour in the project's charts.

## Type

The wordmark is Inter Display, Medium for "Turbo" and Bold for "EXL3", outlined in every file, so no font is needed to
show it. Inter is free under the SIL Open Font License (https://github.com/rsms/inter): use it for text set beside the
logo.

## Use

- Leave at least one cell's width (about a fifth of the mark's height) clear around the logo.
- Keep the lockup at least 120 px wide and the mark at least 16 px.
- Use the light version on light grounds and the dark version on dark grounds.
- Don't recolour, rearrange or stretch the cells: the pattern is the Hadamard matrix.
