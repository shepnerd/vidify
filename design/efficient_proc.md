检查hico，internvl3.5的efficient处理是否纳入，即一些token merging的策略和一些动态的区域tokenization的策略。

hico (https://arxiv.org/pdf/2501.00574): the idea is when extract visual tokens from clips, do token merge on those tokens for token reduction. then in llm processing, do random token dropout. it should be drop-in implementation for existing mllms. for dropout implementation, it may hurt kv cache. don't know anyway to improve it.
\subsection{\shortname: Efficient Long Video Modeling}

To enable MLLMs to handle thousands of input frames, we propose a new video context compression paradigm named \methodname{} (\shortname). This paradigm decomposes video context compression into two main stages: 1. \textbf{Clip-level} Compression during the encoding of long videos. 2. \textbf{Video-level} Compression within the context interaction in the LLM. Based on this framework, we have designed an innovative efficient Video MLLM architecture, VideoChat-Flash, as illustrated in \cref{fig:overview}. Below, we elaborate on our specific design details from data input to model output.


\paragraph{Duration-based Sampling.} First, we need to perform frame sampling on the original video. Specifically, we sample a raw video with a duration of $D$ to obtain $T$ frames as input. Considering that the requirements for understanding short and long videos often differ, we aim to conduct dense sampling on short videos to capture detailed motions and sparse sampling on long videos to focus on event understanding. To this end, we have designed a Duration-based Sampling strategy:
\begin{equation}
    T = \min(T_{\text{max}}, \max{(D, T_{\text{min}})}).
\end{equation}
Simultaneously, we define the sampling density $\phi$ as follows:
\begin{equation}
    \phi(T,D) = \frac{T}{D} = \frac{\min(T_{\text{max}}, \max{(D, T_{\text{min}})})}{D}.
\end{equation}
That is, for short videos where $D < T_{\text{min}}$, $\phi=T_{\text{min}}/D$ , which increases as the video length decreases. For long videos where $D > T_{\text{max}}$, $\phi=T_{\text{max}}/D$, which decreases as the video length increases.

\paragraph{Timestamp Prompt.} For video MLLMs, the ability to perceive timestamps is also a crucial capability. Unlike previous works~\cite{timechat,timesuite} that rely on additional modules or designs to achieve this (there is a considerable computational burden when there are a large number of video frames), we employ a simple timestamp prompt after the video context: \textbf{\textit{“The video lasts for N seconds, and T frames are uniformly sampled from it.”}} We find that this straightforward approach is sufficient to enable the model to perceive the timestamps of the input video, achieving excellent performance on timestamp sensitive tasks such as temporal grounding (see \cref{tab:main}). 

\paragraph{Spatio-Temporal Compression Encoding for Clips.} Considering the substantial redundant and repetitive information, such as that of backgrounds and objects, present between adjacent frames in natural videos, we segment the original video frames into several clips. Subsequently, we employ a video encoder with spatio-temporal attention to encode these clips. This enables each visual token to aggregate information from other frame tokens as comprehensively as possible. Finally, we utilize token merging to combine highly similar tokens. Formally, given a frame sequence sampled from the original video, we divide it into $N_c$ equally sized clips. The frames of $j_{\text{th}}$ clip $\mathbf{x}^j$ are transformed by a video encoder and a connector $\mathcal{F}$, resulting in $M$ compressed visual tokens:
\begin{equation}
    [\mathbf{v}_i^j]_{i=1,2,..,M} = \mathcal{F}(\mathcal{V}(\mathbf{x}^j)),
\end{equation}
where $\mathcal{F}$ consists of a parameter-free similar token merge operation and an MLP projection. Ultimately, we concatenate the compressed tokens of each clip to obtain the input  for the LLM:
\begin{equation}
    \mathbf{X_v}= \text{Concat}([\mathbf{v}_i^1]_{i=1,2,..,M}, \cdot\cdot\cdot, [\mathbf{v}_i^{N_c}]_{i=1,2,..,M}).
\end{equation}

Benefiting from the effectiveness of the video encoder in modeling spatio-temporal interactions, we achieve an extremely heavy compression while well retaining the key information, with each video frame being compressed to an average of only \textbf{16} tokens.

\paragraph{Progressive Visual Dropout in LLM.} Although clip-level compression has been carried out before, due to the possibility of longer-range visual redundancies in long videos (e.g. surveillance videos), and when an LLM responds to specific instructions regarding the visual input, it may not be necessary to continuously focus on the entire long video context. We consider conducting further video-level compression during the LLM inference stage. Recent works~\cite{fastv,llavolta} have explored acceleration strategies for MLLMs when processing short visual contexts. Most of them drop visual tokens based on the correlation between text tokens and visual tokens. In contrast, we find that when the LLM processes a long video context, it pays attention to the entire long video context at the shallow layers of the LLM, while focusing on the details of certain local moments at the deep layers (see the Appendix for specific visualizations). Based on this observation, we have designed a progressive visual dropout strategy, which is divided into two stages. At the shallow layers of the LLM, we uniformly drop a small number of video tokens (i.e. uniform drop), reducing the computation while maintaining the original spatio-temporal structure of the video context. At the deep layers of the LLM, we rely on the correlation between text tokens and video tokens to retain the most critical relevant information (i.e. text-guided select). We have found that this operation not only effectively improves the computational efficiency of the model but also slightly enhances the understanding performance of the model by reducing irrelevant visual noise.

internvl3.5动态token策略 (https://arxiv.org/pdf/2508.18265)：in internvl3.5, it need to learn a router. in practice, maybe we can take some quick implementation. we leverage relative total variation idea (https://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/texturesep12.pdf), that when a patch token rtv is high, we keep more tokens, while rtv is low, we keep fewer tokens. also, this should be drop-in impl.
InternVL3.5-Flash. Compared to InternVL3.5, InternVL3.5-Flash further integrates the Visual Resolution
Router (ViR), thus yielding a series of efficient variants suitable for resource-constrained scenarios. Specifically,
in InternVL3.5, each image patch is initially represented as 1024 visual tokens for the vision encoder, which
are then compressed into 256 tokens via a pixel shuffle module before being passed to the Large Language
Model (LLM). In InternVL3.5-Flash, as shown in Figure 2, an additional pixel shuffle module with a higher compression rate is included, enabling compression of visual tokens down to 64 tokens. For each patch, the
patch router determines the appropriate compression rate by assessing its semantic richness, and routes it to the
corresponding pixel shuffle module accordingly. Benefiting from this patch-aware compression mechanism,
InternVL3.5-Flash is able to reduce the number of visual tokens by 50% while maintaining nearly 100% of the
performance of InternVL3.5, as shown in Section 3.15. Visual Consistency Learning. We further include ViCO as an additional training stage to integrate the visual
resolution router (ViR) into InternVL3.5, thereby reducing the inference cost of InternVL3.5. The obtained
efficient version of InternVL3.5 are termed as InternVL3.5-Flash. In particular, ViCO comprises two stages:
(1) Consistency training: In this stage, the entire model is trained to minimize the divergence between response
distributions conditioned on visual tokens with different compression rates. In practice, we introduce an extra
reference model, which is frozen and initialized with InternVL3.5. Given a sample, each image patch is
represented as either 256 or 64 tokens, and the training objective is defined as follows:
LViCO = Eξ∼R"
1
N
X
N
i=1
KL
πθref (yi
| y<i, I)





 πθpolicy (yi
| y<i, Iξ)

#
, (7)
where KL denotes the KL divergence and ξ denotes the compression rate, which is uniformly sampled from
{
1
4
,
1
16 }. The image Iξ is represented as 256 tokens when ξ =
1
4
and 64 tokens when ξ =
1
16 . We note that the
reference model always performs inference with ξ =
1
4
.
(2) Router training: This stage aims to train the ViR to select an appropriate trade-off resolution for different
inputs. ViR is formulated as a binary classifier and trained using standard cross-entropy loss. To construct the
route targets, we first compute the KL divergence between the model outputs conditioned on uncompressed
visual tokens (i.e., 256 tokens per patch) and those conditioned on compressed visual tokens (i.e., 64 tokens per
patch). During this stage, the main MLLM (ViT, MLP and LLM) is kept frozen, and only the ViR is trained.
Specifically, we first compute the loss ratio for each patch:
ri =
LViCO
yi
| I 1
16

LViCO
yi
| I 1
4
 , (8)
which quantifies the relative increase in loss caused by compressing the visual tokens. Based on this ratio, the
binary ground-truth label for the patch router is defined as:
y
router
i =

0, ri < τ (compression has negligible impact)
1, ri ≥ τ (compression has significant impact),
(9)
where y
router
i = 0 and y
router
i = 1 indicate that the compression rate ξ is set to 1
16 and 1
4
, respectively. During
training, we store the historical ri values of a sliding window, and τ is a dynamical threshold computed
from the k-th percentile of historical ri values. In practice, the target distribution is balanced. During the
consistency training stage, all patches of the same image are represented with a random compression rate, in
order to ensure that the model retains its capability when no compression is applied. As shown in Section 3.15,
InternVL3.5-Flash reduces 50% of the visual tokens while maintaining nearly 100% of the original performance.


slow-fast sampling stragety.
Slow-Fast Video Encoding: for the video encoding with vary FPS, resolutions and duration, linearly
increasing any of these factors would lead to a sharp increase in the token budget on the LLM side,
thus making it challenging to strike a balance between performance and cost. To our knowledge, most
existing MLLMs typically adopt a fixed number of frames and accordingly reduce the resolution of
each frame to meet token budget limitations. Following the paradigm, Qwen-2.5-VL further proposes
2D convolution technique to merge the adjacent frames, aiming to enable the LLM decoder to perceive
more video signals within a fixed frame count. Nevertheless, under the uniform frame sampling
strategy, although many adjacent frames may be highly similar, there can still be some cases where
consecutive frames show significant differences, especially when sampling-interval is larger, a person
is moving or viewpoint is shifting. As a result, the rough 2D convolution merging technique maybe
unfriendly to effective video understanding, since it relies on overly strong assumptions. Considering
the inherent characteristics of video: where adjacent frames are mostly similar yet sometimes significant
changes, we propose a SlowFast video encoding strategy:
– Slow Pathway: This pathway is designed to capture visual information from rapidly changing
frames. It operates at a lower number of frames but with higher resolution.
– Fast Pathway: In contrast, the Fast Pathway captures subtle changes visual signal from relatively
static frames. It uses a higher number of frames but at a lower resolution.
To identify the slow/fast frames from the video, we first devise a patch-based similarity function to
extract them: (1) The first frame is always defined as a slow frame; (2) For each subsequent frame, if its
patch similarity with the latest slow frame exceeds 95%, it is marked as a fast frame; otherwise, it is
marked as a new slow frame. After obtaining the slow and fast frames, we set the fast frame’s token
budget to 30% of a slow frame’s budget to balance the trade-off between frame numbers and the total
token budget. Then, we utilize a binary search technique to precisely calculate the number of tokens
per slow frame under the total token budget limitation (e.g., 75,000 tokens in Keye-VL-1.5). Meanwhile,
to more clearly identify the boundaries and timestamp information between Slow and Fast frames,
6
we introduce additional special tokens along with absolute timestamps to guide the model during
learning, as shown in Figure 3.