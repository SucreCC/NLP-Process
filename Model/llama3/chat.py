import fire
import time
import torch

from generation import Llama


def interactive_inference(
        ckpt_dir: str = "../../others/pretrain-model/llama/Llama3.1-8B",
        tokenizer_path: str = "../../others/pretrain-model/llama/Llama3.1-8B/tokenizer.model",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 32,
):
    # 初始化模型
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,  # 单次处理一个输入
        flash=False,  # 禁用闪存注意力以确保一致性
    )

    # 提示用户如何退出
    print("欢迎使用 LLaMA 模型交互式服务！")
    print("输入 'exit' 或按 Ctrl+C 结束会话。\n")

    # 交互循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1337)

    while True:
        try:
            # 获取用户输入
            user_input = input("请输入提示词: ")
            if user_input.strip().lower() == "exit":
                print("结束会话。再见！")
                break

            # 记录时间
            t0 = time.time()

            # 调用模型生成
            results = llama.text_completion(
                [user_input],  # 单次处理用户输入
                sample_rng=sample_rng,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            t1 = time.time()
            print(f"\n生成完成，用时 {t1 - t0:.2f} 秒")
            print("生成结果:")
            print(results[0]["generation"])
            print("\n==================================\n")

        except KeyboardInterrupt:
            print("\n会话已中断。再见！")
            break


if __name__ == "__main__":
    fire.Fire(interactive_inference)