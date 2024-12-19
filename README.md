# Showcase (work in progress)
Here you can find a curated collection of
  1. code that showcases my skills and style, and
  2. products that I'm proud to have worked on.

Although my [Coding Showcase](https://github.com/MelleStarke/Showcase/Code) and [Product Showcase](https://github.com/MelleStarke/Showcase/Products) both have their own pages, below you will find some examples of each.

## Code

### Kenku
Kenku is the product I'm developing for my Master's Thesis. Though I don't want to spoil the whole thing, I would like to showcase a recurring module in my network.
This module is based on the work of [Kameoka et al. (2021)](https://arxiv.org/pdf/2104.06900).

```python
class KameBlock(nn.Module):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               out_ch: int,
               embed_ch: int,
               num_classes: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               num_output_streams: Optional[int] = 1,
               signal_segment_len: int = 80,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.1,
    ):
    """Class for recurring structure in FastConvS2S-VC model. Based on Kameoka et al.'s model.
       Features:
         - Linear input and output layers.
         - Several 1D, causal, dilated convolutional layers followed by GLUs, with skip connections.
         - Concatenated class embeddings at every layer (linear and conv).
         - Dynamic padding, depending on kernel size and dilation. 
           Copied from previous input if possible, such that it temporally overlaps.

    Args:
        in_ch (int): nr. of features in the input signal
        conv_ch (int): nr. of convolutional channels (uniform across conv layers)
        out_ch (int): nr. of features in the output signal
        embed_ch (int): nr. of features used to embed classes
        num_classes (int): nr. of classes
        num_layers (int, optional): nr. of convolutional layers. Defaults to 8.
        kernel_size (int, optional): kernel size. Defaults to 5.
        num_output_streams (int, optional): nr of output streams. Multiplies the out_ch by this factor.
                                            Useful for getting keys and values from a single block, for example. Defaults to 1.
        signal_segment_len (int, optional): length of the segments that the input signal is broken up in. Defaults to 80.
        dilations (List[int], optional): list of dilations in the conv layers. Defaults to [1, 3, 9, 27, 1, 3, 9, 27].
        dropout_rate (float, optional): drop-out rate. Defaults to 0.1.
    """
    super(KameBlock, self).__init__()
    
    self.num_output_streams = num_output_streams
    # Keep track of the paddings each conv layer outputs. To be used during the next forward call.
    self.paddings = [None] * num_conv_layers
    
    if dilations is None:
      dilations = [3**(i%4) for i in range(num_conv_layers)]
      
    else:
      assert len(dilations) == num_conv_layers,\
        f"The manually provided list of dilations has a different length ({len(dilations)}) than the nr. of layers ({num_conv_layers})."
    
    #=== Layers ===#
    self.dropout     = nn.Dropout(p=dropout_rate)
    self.embed_layer = nn.Embedding(num_classes, embed_ch)  # Embedding of class values into class feature space.
                                                            # TODO: original src code has weight norm commented out. Give a try?
    
    # Use 1D Conv layer as linear layer to match up shapes better.
    self.in_layer = weight_norm(nn.Conv1d(in_channels  = in_ch + embed_ch,
                                          out_channels = conv_ch,
                                          kernel_size  = 1, padding = 0))
    
    self.conv_blocks = nn.ModuleList()
    for dil in dilations:
      self.conv_blocks.append(ConvGLU(in_ch       = conv_ch + embed_ch, 
                                      out_ch      = conv_ch, 
                                      kernel_size = kernel_size, 
                                      dilation    = dil))
    
    self.out_layer = weight_norm(nn.Conv1d(in_channels  = conv_ch + embed_ch,
                                           out_channels = out_ch * num_output_streams,
                                           kernel_size  = 1, padding = 0))
  
  def forward(self, X: Tensor, class_id: Union[int, List[int]]):
    device = X.device
    batch_size, in_ch, timesteps = X.shape
    
    if len(np.shape(class_id)) == 0:
      class_id = [class_id]
    
    # Take class id list, add empty dim with unsqueeze, broadcast over timesteps.
    # Prepration for embedding layer pass and appending to input.
    class_tensor = torch.LongTensor(class_id).unsqueeze(1).repeat(1, timesteps)
    embedding    = self.embed_layer(class_tensor).permute(0, 2, 1)
    
    #=== Forward Pass ===#
    X_    = self.dropout(X)
    X_emb = concat_embedding(embedding, X_)
    X_    = self.in_layer(X_emb)
    
    # Pass through Conv GLU blocks.
    for i, layer in enumerate(self.conv_blocks):
      X_, padding = layer(X_, embedding, padding=self.paddings[i])
      self.paddings[i] = padding
    
    X_emb = concat_embedding(embedding, X_)
    Y = self.out_layer(X_emb)
    
    if self.num_output_streams > 1:
      # Split into multiple matrices of shape (batch_size, out_ch, timesteps).
      # For use in (K, V) split for attention module, and (Delta, Sigma, Phi) split in attention predictor
      Y_streams = torch.chunk(Y, chunks=self.num_output_streams, dim=1)
      return Y_streams  # Formatted as a tuple of tensors with equal feature/channel dims.
    
    return Y
  
  def clear_paddings(self):
    """
    Reset dynamic padding to None. Used between passing of batches.
    """
    self.paddings = [None] * len(self.conv_blocks)
  
  
class ConvGLU(nn.Module):
  """1D Convolutional GLU block. Features:
      - Speaker embedding
      - Causal "look-behind" padding (i.e. tail of the previous forward call's input)
      - Dilation
      - GLU activation (base activation function: sigmoid)
      - Skip-connection
      - Weight normalization
  """
  def __init__(self,
               in_ch: int, 
               out_ch: int, 
               kernel_size: int, 
               dilation: int): 
    super(ConvGLU, self).__init__()
      
    self.padding_len = calc_padding(kernel_size, dilation)
    self.conv = nn.Conv1d(
      in_ch,
      out_ch * 2,  # Double output size for GLU (for splitting into two matrices)
      kernel_size,
      dilation = dilation,
      padding = 0
    )
    self.conv = weight_norm(self.conv)
    
  def forward(self, X: Tensor, embedding: Tensor, padding: Tensor = None):
    # TODO: maybe fiddle with doing embedding or padding first.
    # Source authors do input > embedding > padding > forw. pass > sum input and output
    X_emb = concat_embedding(embedding, X)
    
    # Pad sequence with zeros if no padding is passed.
    if padding is None:
      # Make tensor of zeros of shape (batch_size, channels) and append empty dim at the end.
      padding = torch.zeros_like(X_emb[:,:,1]).unsqueeze(-1)
      padding = padding.repeat(1, 1, self.padding_len)  # Repeat zeros over empty dim for padding.
      
    
    # Append along time dimension, prior to input X.
    X_emb_pad = torch.cat([padding, X_emb], dim=-1)
    X_        = self.conv(X_emb_pad)
    
    with torch.no_grad():  # TODO: not sure if no_grad() is useful/valid. Maybe try without.
      padding = X_emb_pad[:,:,-self.padding_len:]
      
    X_scale, X_gate = torch.chunk(X_, chunks=2, dim=1)  # Split into 2 streams for GLU
    Y = X_scale * torch.sigmoid(X_gate)  # GLU
    Y += X  # Skip-connection
    return Y, padding
```
