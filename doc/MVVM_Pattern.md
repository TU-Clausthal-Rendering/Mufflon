# Model View ViewModel Pattern

MVVM divides the application into three layers:
* model
* view
* viewModel

### Model
The model refers to the domain model (data and it's behaviour). A single model class exposes an interface which ensures that the model is always in a valid state.

### View
The view describes what the user sees on the screen (here XAML description). Views contain no logic and only forward events to the underlying view model.

### ViewModel
The view model is the adapter from model to view. It transforms the model data to make it compatible with respective the view. 

### Commands
Some views require `ICommand` bindings. They generally do two things:
* Execute() will be called if the command **is** executed (after button click)
* CanExecute() indicates if the command **can be** executed

Commands change the underlying model (and sometimes the view model) and display new views in forms of dialogs.

## General Rules

* The **view** has no references to its **view model**. After the views creation, the corresponding view model is set via: `view.DataContext = viewModel`
* The **model** has no references to any **view model**.
* Changes from **model** to **view model** or **view** to **view model** are implemented with events with the `INotifyPropertyChanged` interface.
* The application should be fully functional without any view model (all logic that is required for the application resides inside the **model**). 
* The **view** (xaml) has (almost) no code-behind.
* **View models** should not have refences to other view models. The only exception are parent/child references for lists and list elements.

## Indications that something went wrong

* A function from one **view model** is required by another view model
    * The function probably belongs into a model
* A **command** from one **view model** is required by another view model
    * The functionality from the command probably probably belongs into a model
* The **view** has code-behind
    * Try to put the code into the undelying view model
    * If the view has code that modifies the behaviour of one UI-element change the behaviour of the UI-element instead (i.e. with inheritance)