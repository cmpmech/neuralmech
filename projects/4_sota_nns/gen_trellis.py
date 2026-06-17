from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

outputs = pipeline.run(image)
