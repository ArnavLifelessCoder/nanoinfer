# M1 CI regression history

| Run | Failure | Fix |
| --- | --- | --- |
| [39420114195](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39420114195) | All four pytest jobs failed because the basetemp parent was absent in a clean checkout. | Use the runner's existing temporary directory. |
| [39420544251](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39420544251) | GCC Debug and ASan rejected copied structured bindings under `-Werror=range-loop-construct`. | Bind metadata cases by const reference. |
| [39420544251](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39420544251) | Windows CTest selected system Python, which could not import NumPy. | Explicitly select the project venv. |
| [39421570280](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39421570280) | All five required jobs passed on the previous review head, `dd3d5ba`. UBSan recovery was subsequently identified as a gap in the gate. | Add a deliberate-overflow regression and disable sanitizer recovery. |
| [39423684527](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39423684527) | On `9877dc8`, the new Linux ASan regression failed: `UBSan reported signed overflow but exited successfully`. The other four CTest entries passed. | Add `-fno-sanitize-recover=all` to both compile and link options. |
| [39424010295](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39424010295) | The log-sanitization test found a duplicate dictionary key when the actual project directory matched the hardcoded Windows runner checkout path. | Remove the redundant hardcoded project key; the detected project root already covers it. |

These links replace the four raw hosted log dumps. Complete logs remain in the
linked runs. Local Debug and ASan test output remains in this directory.
