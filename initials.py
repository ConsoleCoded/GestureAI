# signature.py
def signature():
    print('GestureAI by Abhishek Kumar Singh (Consolecoded)')

def enforce_signature():
    import builtins
    old_exit = builtins.exit
    def custom_exit(*args, **kwargs):
        print("👋 Thanks for using GestureAI! - AKS")
        old_exit(*args, **kwargs)
    builtins.exit = custom_exit

signature()
enforce_signature()