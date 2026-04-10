from pipeline_manager import PipelineManager

if __name__ == "__main__":
    app = PipelineManager(
        det_p="models/best.pt",
        cls_p="models/best_large.pt",
        cat_p="product_catalog_sut.json",
        test_d="test_images",
        baraj=0.25
    )
    app.run()