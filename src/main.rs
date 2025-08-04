use anyhow::Result;
use tch::{Device, Tensor, Kind, CModule};
use tokenizers::Tokenizer;

pub struct Specter2 {
    model: CModule,
    tokenizer: Tokenizer,
    device: Device,
}

impl Specter2 {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let device = Device::cuda_if_available();
        Ok(Self {
            model: CModule::load_on_device(model_path, device)?,
            tokenizer: Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?,
            device,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{}", e))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let input_ids = Tensor::from_slice(&input_ids).to_device(self.device).unsqueeze(0);
        let attention_mask = Tensor::from_slice(&attention_mask).to_device(self.device).unsqueeze(0);
        Ok(self.model.forward_ts(&[input_ids, attention_mask])?)
    }
}

// Usage:
fn main(){
    let model = Specter2::new(
    "/Users/ryanhammonds/projects/neurovlm/src/neurovlm/neurovlm_data/specter2_traced.pt",
    "/Users/ryanhammonds/projects/neurovlm/src/neurovlm/neurovlm_data/tokenizer/tokenizer.json"
    ).unwrap();
    let embedding = model.encode("testing").unwrap();
    let first_10 = embedding.flatten(0, -1).slice(0, 0, 10, 1);
    let values: Vec<f64> = first_10.try_into().unwrap();
    println!("First 10 values: {:?}", values);
}
