```bash
ilab config init
ilab model download
```

```bash
ilab model serve
```

```bash
ilab model chat
```

## QNA
https://www.youtube.com/watch?v=snMUJGXozec

```bash
mkdir ~/.local/share/instructlab/taxonomy/knowledge/test
cp qna.yaml ~/.local/share/instructlab/taxonomy/knowledge/test
ilab taxonomy diff
```
The below commands are not working [due to the bug](https://github.com/instructlab/instructlab/issues/3530).
```bash
ilab data generate --num-instructions 10
ilab model train --pipeline simple
```
