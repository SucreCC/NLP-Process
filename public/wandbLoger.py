from wandb.apis.importers import wandb
import wandb
import os


class WandbLogger:
    def __init__(self, project_name, api_key=None, run_name=None, config=None):
        """
        初始化 WandB Logger
        :param project_name: 项目名称
        :param api_key: WandB 的 API Key（如果已经配置到环境变量中可不传）
        :param run_name: 本次运行的名称
        :param config: 配置字典（超参数等）
        """
        # 设置 API Key
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        # 登录 WandB
        wandb.login()

        # 初始化 WandB 项目
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )

    def log(self, metrics):
        """
        日志记录
        :param metrics: 字典形式的日志数据
        """
        wandb.log(metrics)

    def save_model(self, model_path):
        """
        保存模型文件到 WandB
        :param model_path: 模型文件路径
        """
        wandb.save(model_path)

    def add_text(self, key, text):
        """
        添加文本日志
        :param key: 文本的键名
        :param text: 文本内容
        """
        wandb.log({key: text})

    def add_image(self, key, image, caption=None):
        """
        添加图像日志
        :param key: 图像的键名
        :param image: 图像数组或路径
        :param caption: 图像说明
        """
        wandb.log({key: [wandb.Image(image, caption=caption)]})

    def finish(self):
        """
        结束当前 WandB 运行
        """
        wandb.finish()



# 使用示例
if __name__ == "__main__":

    # 初始化 WandB Logger
    logger = WandbLogger(
        project_name="BERT-Chinese-Classification",

        # api_key= os.environ.get("WANDB_API_KEY"),  # 如果已经设置了环境变量，可以省略
        run_name="experiment_1",
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
        }
    )

    # 记录训练过程
    for epoch in range(10):
        logger.log({"epoch": epoch, "loss": 0.01 * (10 - epoch), "accuracy": 0.1 * epoch})

    # 保存模型
    logger.save_model("model.h5")

    # 添加文本日志
    logger.add_text("example_text", "This is a sample text log.")

    # 添加图像日志
    # logger.add_image("example_image", "path/to/image.png", caption="Sample Image")

    # 结束运行
    logger.finish()
