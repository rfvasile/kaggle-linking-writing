((python-base-mode
  . ((python-shell-interpreter . "docker")
     (python-shell-prompt-detect-failure-warning . nil)
     (python-shell-completion-native-enable . nil)
     ;; Pre-fill `SPC d d' with the Docker debugpy config.
     (dape-command . (docker-python))
     (eval . (setq-local
              python-shell-interpreter-args
              (let ((name (car (split-string
                                (shell-command-to-string
                                 "docker ps --filter name=kaggle-notebooks --format '{{.Names}}' 2>/dev/null")))))
                (format "exec -i -w /kaggle/working -e PYTHONUNBUFFERED=1 -e PYTHONSTARTUP=/kaggle/working/scripts/comint_mpl_show.py %s sh -c \"exec python -i 2>&1\""
                        (if (and name (string-prefix-p "kaggle-notebooks" name))
                            name
                          "kaggle-notebooks-gpu")))))))
 ;; Rewrite container traceback paths before compile.el opens them.
 (inferior-python-mode
  . ((eval . (let ((root (file-name-as-directory default-directory)))
               (when (file-directory-p (concat root ".venv/kaggle-source"))
                 (setq-local
                  compilation-transform-file-match-alist
                  (append
                   `(("\\`/kaggle/working/" ,root)
                     ("\\`/kaggle/input/" ,(concat root "input/"))
                     ;; Package roots are flattened into one mirror.
                     ("\\`\\(?:/usr\\(?:/local\\)?\\|/root/\\.local\\)/lib/python[0-9.]+/\\(?:dist\\|site\\)-packages/"
                      ,(concat root ".venv/kaggle-source/"))
                     ;; Keep stdlib below package roots; first match wins.
                     ("\\`/usr\\(?:/local\\)?/lib/python[0-9.]+/"
                      ,(concat root ".venv/kaggle-stdlib/"))
                     ;; Drop container-only scratch paths.
                     ("\\`/tmp/" nil))
                   compilation-transform-file-match-alist))))))))
