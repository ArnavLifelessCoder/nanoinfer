#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <dbgeng.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace {
using Microsoft::WRL::ComPtr;

class DebugOutput final : public IDebugOutputCallbacks {
 public:
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** object) override {
    if (object == nullptr) return E_POINTER;
    *object = nullptr;
    if (id != __uuidof(IUnknown) && id != __uuidof(IDebugOutputCallbacks)) return E_NOINTERFACE;
    *object = static_cast<IDebugOutputCallbacks*>(this);
    AddRef();
    return S_OK;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const auto remaining = --references_;
    if (remaining == 0) delete this;
    return remaining;
  }
  HRESULT STDMETHODCALLTYPE Output(ULONG, PCSTR text) override {
    std::fputs(text, stdout);
    std::fflush(stdout);
    return S_OK;
  }

 private:
  std::atomic<ULONG> references_{1};
};

void check(HRESULT result, const char* operation) {
  if (FAILED(result)) {
    std::fprintf(stderr, "%s failed: 0x%08lx\n", operation, static_cast<unsigned long>(result));
    throw std::runtime_error(operation);
  }
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc != 3) {
    std::fputs("dump_stacks_windows DUMP SYMBOL_PATH\n", stderr);
    return 2;
  }
  try {
    const std::filesystem::path dump(argv[1]);
    ComPtr<IDebugClient5> client;
    check(DebugCreate(__uuidof(IDebugClient5), reinterpret_cast<void**>(client.GetAddressOf())),
          "DebugCreate");
    ComPtr<IDebugControl> control;
    ComPtr<IDebugSymbols3> symbols;
    check(client.As(&control), "IDebugControl");
    check(client.As(&symbols), "IDebugSymbols3");
    ComPtr<DebugOutput> output;
    output.Attach(new DebugOutput);
    check(client->SetOutputCallbacks(output.Get()), "SetOutputCallbacks");
    check(symbols->SetSymbolPathWide(argv[2]), "SetSymbolPathWide");
    check(client->OpenDumpFileWide(dump.c_str(), 0), "OpenDumpFileWide");
    check(control->WaitForEvent(0, INFINITE), "WaitForEvent");
    check(control->Execute(DEBUG_OUTCTL_THIS_CLIENT, ".reload; ~*kb; lm", DEBUG_EXECUTE_DEFAULT),
          "Execute stacks");
    check(client->EndSession(DEBUG_END_PASSIVE), "EndSession");
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}
