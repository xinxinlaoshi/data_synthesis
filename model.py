from openai import OpenAI
import argparse
import json

class ChatGPT():
    def call_gpt(self, data, model = "gpt-4o-mini", temperature = 1, response_format = "json_object"):
        client = OpenAI()
        if temperature < 0 or temperature > 2:
            raise Exception("The range of temperature should be between 0 and 2")

        if response_format == "json_object" or response_format == "text":
            response_format = {"type": response_format}
            completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format=response_format,
            messages=[
                {"role": "system", "content": data["prompt"]},
                {"role": "user", "content": data["user_input"]}
            ]
            )
            return completion.choices[0].message.content
        else:
            raise Exception("Response format is either text or json")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="gpt-4o-mini")
    parser.add_argument("--temperature", 
                        type=int,
                        default=1)
    parser.add_argument("--response_format", 
                        type=str,
                        default="json_object")
    parser.add_argument("--prompt_dir", 
                        type=str,
                        default="")
    # 只有单条 input 的输入
    parser.add_argument("input", 
                        type=str,
                        required=True)
    parser.add_argument("output_dir",
                        type=str,
                        required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = ChatGPT()

    with open(args.prompt_dir, "r") as f:
        prompt = f.read()
        
    data = {"prompt": prompt, "user_input": args.input}
    result = model.call_gpt(data, args.model, args.temperature, args.response_format)

    if args.response_format == "text":
        with open(args.output_dir, "a") as f:
            f.write(result)
    
    if args.response_format == "json_object":
        result = json.loads(result)
        with open(args.output_dir, "a") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    