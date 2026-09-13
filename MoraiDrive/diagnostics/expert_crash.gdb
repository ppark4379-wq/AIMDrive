set pagination off
set confirm off
set print thread-events off
set disable-randomization off
handle SIGPIPE nostop noprint pass
handle SIGFPE stop print pass
handle SIGSEGV stop print pass
handle SIGABRT stop print pass
handle SIGILL stop print pass
handle SIGBUS stop print pass
run
if $_isvoid($_exitcode)
  echo \n=== NATIVE FAULT DETAILS ===\n
  p $_siginfo
  info symbol $pc
  x/10i $pc
  info registers rip rax rdi rsi mxcsr fctrl fstat xmm0 xmm1
  thread apply all bt 12
  info sharedlibrary
  handle SIGFPE nostop print pass
  handle SIGSEGV nostop print pass
  handle SIGABRT nostop print pass
  handle SIGILL nostop print pass
  handle SIGBUS nostop print pass
  continue
end
