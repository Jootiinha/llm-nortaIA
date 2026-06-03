# Metrics

Este projeto agora registra metricas de desempenho do treino em TensorBoard a partir de [scripts/train_model.py](/Users/joaocrm/Documents/dev/estudos/llm-nortaIA/scripts/train_model.py). As configuracoes da run ficam em `configs/models/*.yaml`.

## Como usar

Instale as dependencias:

```bash
make install
```

Escolha uma configuracao, por exemplo `joao-gpt-mini-v1`, e rode o treino:

```bash
make train_model model_name=joao-gpt-mini-v1
```

Em outro terminal, abra o TensorBoard:

```bash
make tensorboard
```

Logs gerados:

```text
checkpoints/<nome-do-modelo>/tb
```

Exemplos de configs:

- [joao-gpt-mini-v1.yaml](/Users/joaocrm/Documents/dev/estudos/llm-nortaIA/configs/models/joao-gpt-mini-v1.yaml)
- [joao-gpt-mini-v2.yaml](/Users/joaocrm/Documents/dev/estudos/llm-nortaIA/configs/models/joao-gpt-mini-v2.yaml)

## Metricas registradas

### Preparacao de dados

- `prep/tokenization_seconds`: tempo total de tokenizacao do dataset bruto.
- `prep/grouping_seconds`: tempo total para agrupar tokens em blocos fixos.
- `prep/raw_train_examples`: quantidade de exemplos brutos de treino.
- `prep/raw_validation_examples`: quantidade de exemplos brutos de validacao.
- `prep/train_blocks`: quantidade final de blocos de treino.
- `prep/validation_blocks`: quantidade final de blocos de validacao.
- `prep/train_tokens`: total teorico de tokens de treino apos agrupamento.
- `prep/validation_tokens`: total teorico de tokens de validacao apos agrupamento.

### Configuracao do modelo e run

- `model/num_parameters`: numero total de parametros.
- `model/vocab_size`: tamanho do vocabulario.
- `model/block_size`: tamanho de contexto por bloco.
- `model/n_layer`: numero de camadas.
- `model/n_head`: numero de heads.
- `model/n_embd`: dimensao dos embeddings.
- `run/batch_size`: batch size por device.
- `run/eval_batch_size`: batch size de avaliacao por device.
- `run/gradient_accumulation_steps`: acumulacao de gradiente.
- `run/effective_examples_per_step`: exemplos efetivos por step de otimizacao.
- `run/effective_tokens_per_step`: tokens efetivos por step de otimizacao.

### Throughput

- `perf/step_time_sec`: tempo aproximado do step atual.
- `perf/steps_per_sec_window`: steps por segundo na janela entre logs.
- `perf/examples_per_sec_window`: exemplos por segundo na janela entre logs.
- `perf/tokens_per_sec_window`: tokens por segundo na janela entre logs.
- `perf/tokens_per_step`: tokens teoricos por step.
- `perf/examples_per_step`: exemplos teoricos por step.

### Uso de CPU e memoria

- `system/process_cpu_percent`: percentual de CPU consumido pelo processo Python.
- `system/process_rss_mb`: memoria RSS do processo em MB.
- `system/process_ram_percent`: percentual da RAM total consumido pelo processo.
- `system/process_threads`: numero de threads do processo.

### GPU

Se estiver usando CUDA:

- `system/gpu_memory_allocated_mb`
- `system/gpu_memory_reserved_mb`
- `system/gpu_memory_max_allocated_mb`

Se estiver usando MPS no Apple Silicon e o backend expuser essas informacoes:

- `system/mps_current_allocated_mb`
- `system/mps_driver_allocated_mb`
- `system/mps_recommended_max_memory_mb`

### Avaliacao

- `eval/loss`: emitido pelo `Trainer`.
- `eval/perplexity`: calculado manualmente a partir de `eval_loss`.

## Como interpretar

- `tokens_per_sec_window` e `steps_per_sec_window` sao as metricas mais importantes para medir throughput real.
- `process_rss_mb` mostra o crescimento de memoria do processo Python ao longo do treino.
- `gpu_memory_max_allocated_mb` ajuda a identificar se ainda ha margem para aumentar batch size em CUDA.
- `tokenization_seconds` e `grouping_seconds` mostram se o gargalo esta na preparacao dos dados antes do treino.

## Dicas praticas de otimizacao

- Se `tokens_per_sec_window` estiver baixo e `process_cpu_percent` estiver alto, o gargalo tende a estar em CPU, tokenizacao, dataloader ou preprocessing.
- Se `gpu_memory_allocated_mb` estiver baixa e `steps_per_sec_window` tambem estiver baixa, a GPU pode estar subutilizada.
- Se `process_rss_mb` crescer demais durante `dataset.map`, reduza paralelismo ou reveja o pipeline de tokenizacao.
- Compare runs sempre com o mesmo dataset, `block_size` e tamanho de modelo. Caso contrario, o throughput deixa de ser comparavel.

## Observacao

No backend `mps`, o nivel de telemetria de memoria e menor do que em CUDA. Nesse caso, use o TensorBoard para throughput e RAM do processo, e complemente com o Monitor de Atividade do macOS para acompanhar GPU.
