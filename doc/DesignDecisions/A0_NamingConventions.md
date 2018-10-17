Classes
-

- CamelCase without prefix
- Either none abstract (pure virtual) or all -> interface, denoted with 'I' prefix
- Member variables are allowed in interfaces
- struct only for POD types, otherwise class

Function and method names
-

- snake_case without prefix
- Indentation of parameter list to opening bracket. When tab indent doesn't exactly match the bracket, fill up with spaces. Whether to break the parameter list at all is up to the programmer's discretion. Example:

        void test(int foo,
                  int bar);


Variables
-

- lowerCamelCase
- (Class) member variables (non-public) get prefixed with 'm_'
- Constants as UPPER_SNAKE_CASE
- Static non-const has 's_' as prefix
- Parameters have same convention as normal variables
- Value parameters are to be treated as const, ie. they must not be changed during the function, regardless whether they are declared as such

Switch-statement:
-

- TODO

Tabs vs. spaces
-

- Tabs with size of 4 spaces
- Prefer tabs over spaces unless needed for alignment
- public, private, ... without extra indent
- Namespace without extra indent
- Otherwise one extra indent per block (function, loop, etc.)

Brackets & whitespaces
-

- Curly brackets open on the same line as their expression, e.g.
        
        for(...) {
- No whitespaces between round brackets and their expression, but between round and curly brackets
- Whitespace between operator and operand (may be omitted at programmer's discretion, e.g. x*x + ...)
- Whitespace after comma in parameter lists
- Empty scopes may have curly brackets right after each other: {}
- Single-line if/loops may omit curly brackets. However, do NOT do

        if(...)
            ...
        else {
            ...
        }
- Else-statement stays on same line:

        if(...) {
            ...
        } else {
            ...
        }

East-side vs. west-side
-

- TODO

Constructor
-

- Double-dot on same line
- Initializers starting on new line, one initialization per line only
- Initializers get extra indentation
- Curly bracket goes into new line without indentation, e.g.

        C() :
            m_a()
        {
            ...
        }

Namespaces
-

- Nested namespaces only as
    
        namespace a::b::c {
- TODO: using namespace std?
- TODO: nested namespaces are c++17 feature - CUDA doesn't support this yet

Auto
-

- auto only for complex types (within limits up to programmer's discretion)
- Deducted pointer types must be made explicit, i.e.

        auto *a = ...