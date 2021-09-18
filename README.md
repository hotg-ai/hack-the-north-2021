# **Goal**

Build a rune using a TensorFlow Lite Machine Learning Model.

# **What we'll be using**

- hotg-rune-cli
- TensorFlow Lite
- YAML

# **Workshop**

- Install & Setup hotg-rune-cli
- Download/Create TFLite model
- Build ML pipeline
- Generate & Test the rune

# **Overview**

### Example - How to perform artistic style transfer? Use ML!

![Dog](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8bb7ec17-4889-425e-921a-3fd1c04a13c2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210918%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210918T020747Z&X-Amz-Expires=86400&X-Amz-Signature=5ccc110ea0ea54e1a5e0f0cc723432782fe23fe8e75f89088be26f1d15fb927a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

### The Neural Style Transfer Model

![Content vs Style Image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/789e6939-4dcb-4c5f-9ac4-8a17ffae273d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210918%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210918T020904Z&X-Amz-Expires=86400&X-Amz-Signature=8fe9eae9587abca30a9fd169c1aeb7e7ab61aa09af4466aa6afe5c1500af698b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

![Neural Style Transfer Model](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fee7dbcf-476a-424b-9d3a-74a84a9c41f4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210918%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210918T020932Z&X-Amz-Expires=86400&X-Amz-Signature=8ab6d29936cfaf0c6c7bf527e5d536a01bbc09cb079bbbd0f1899ca60e907b27&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## Installation

### Installation from Source (Not Recommended)

The **Rune** command line tool builds the **Runefile** into Rust code - need to install the [rust compiler](https://rustup.rs/).

```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install the rune command line tool

```bash
$ cargo install --git https://github.com/hotg-ai/rune rune
```

## Using Rune Command Line Tool

Observe model details using the hotg-rune-cli

```bash
$ rune model-info style_transform.tflite
Ops: 295
Inputs:
	content_image: Float32[1, 384, 384, 3] # Input from content image
	mobilenet_conv/Conv/BiasAdd: Float32[1, 1, 1, 100] # Input from style_predict tflite model
Outputs:
	transformer/expand/conv3/conv/Sigmoid: Float32[1, 384, 384, 3]
```

**Create Runefile.yml**

```yaml
image: runicos/base
version: 1

pipeline:
  content_image:
    capability: IMAGE
    outputs:
    - type: u8
      dimensions: [1, 384, 384, 3]
    args:
      source: 0
      pixel-format: "@PixelFormat::RGB"
      width: 384
      height: 384

  style:
    capability: IMAGE
    outputs:
    - type: u8
      dimensions: [1, 256, 256, 3]
    args:
      source: 1
      pixel-format: "@PixelFormat::RGB"
      width: 256
      height: 256

  normalized_content_image:
    proc-block: "hotg-ai/rune#proc-blocks/image-normalization"
    inputs:
    - content_image
    outputs:
    - type: f32
      dimensions: [1, 384, 384, 3]

  normalized_style_image:
    proc-block: "hotg-ai/rune#proc-blocks/image-normalization"
    inputs:
    - style
    outputs:
    - type: f32
      dimensions: [1, 256, 256, 3]

  style_vector:
    model: ./style_predict.tflite
    inputs:
    - normalized_style_image
    outputs:
    - type: f32
      dimensions: [1, 1, 1, 100]

  style_transform:
    model: ./style_transform.tflite
    inputs:
    - normalized_content_image
    - style_vector
    outputs:
    - type: f32
      dimensions: [1, 384, 384, 3]

  serial:
    out: SERIAL
    inputs:
    - style_transform
```

**Compile the Rune**

```bash
$ rune build Runefile.yaml
```

# Quick Links

- [Documentation](https://hotg.dev/)
- [Build, Run, Serve a Rune](https://hotg.dev/docs/)
- Mobile Application Links
    - [iOS](https://apps.apple.com/us/app/runic-by-hotg-ai/id1550831458)
    - [Android](https://play.google.com/store/apps/details?id=ai.hotg.runicapp&hl=en_US&gl=US)
    - [PWA](https://runicjs.web.app/?utm_source=hotg.dev&utm_medium=web&utm_campaign=CTA)
- [TensorFlow Lite Model Repository](https://tfhub.dev/s?deployment-format=lite)
- [Rune Examples](https://github.com/hotg-ai/rune/tree/master/examples)
- [Rune Repo](https://github.com/hotg-ai/test-runes)
- [Proc-Block Examples](https://github.com/hotg-ai/rune/tree/master/proc-blocks)

# Judging Criteria

1. Overall Application Idea
2. Runefile and Model selection
3. Does the Rune work?
4. Integration of the Rune into the Application

# Rune Submissions

1. Create pull request
2. Upload rune, Runefile & TFLite model
3. Win our sponsorship prize
