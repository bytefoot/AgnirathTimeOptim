N = 2  # temp code

# Bounds for the velocity
Bounds = [(0, 0)] + [(0, 30)]*(N-2) + [(0, 0)]

# List of constraints
Constraints = []

def add_constraint(*, type: str, fun=None):
    # Decorator to add functions to constraints

    if fun:
       Constraints.append({
           "type": type,
            "fun": fun
       })

    else:
        def wrapper(fun):
            Constraints.append({
                "type": type,
                "fun": fun
            })
            
            return fun
        
        return wrapper

# # Demo application
@add_constraint(type='eq')
def constraint1(p):
    print("Hello World", p)

